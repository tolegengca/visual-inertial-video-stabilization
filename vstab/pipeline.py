from __future__ import annotations

import csv, json, shutil, subprocess, tempfile
from pathlib import Path
import cv2
import numpy as np

from .flow import FlowEstimator
from .motion import (accumulate, compensation, consistency, estimate_similarity,
                     grid_sample, jitter_rms, safe_center_crop, smooth_poses)


def feature_fallback(a, b, threshold=2.5):
    ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    p = cv2.goodFeaturesToTrack(ga, 1000, .01, 8)
    if p is None: return None, 0.
    q, st, _ = cv2.calcOpticalFlowPyrLK(ga, gb, p, None)
    good = st.ravel().astype(bool)
    m, ratio, _ = estimate_similarity(p.reshape(-1,2)[good], q.reshape(-1,2)[good], threshold)
    return m, ratio


def _video_info(path):
    cap=cv2.VideoCapture(str(path))
    fps=cap.get(cv2.CAP_PROP_FPS)
    w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps,w,h,count


def _iter_frames(path):
    cap=cv2.VideoCapture(str(path))
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            yield frame
    finally:
        cap.release()


def _motion_diagnostic(frames):
    vals=[]
    for a,b in zip(frames[:-1],frames[1:]):
        m,r=feature_fallback(a,b,3.0)
        if m is not None: vals.append([m[0,2],m[1,2],np.arctan2(m[1,0],m[0,0]),r])
    x=np.asarray(vals)
    if len(x)==0: return {"translation_motion_rms":None,"translation_change_rms":None}
    return {
      "translation_motion_rms":float(np.sqrt(np.mean(np.sum(x[:,:2]**2,axis=1)))),
      "translation_change_rms":float(np.sqrt(np.mean(np.sum(np.diff(x[:,:2],axis=0)**2,axis=1)))) if len(x)>1 else 0.,
      "rotation_motion_rms":float(np.sqrt(np.mean(x[:,2]**2))),
      "mean_tracking_inlier_ratio":float(x[:,3].mean())}


def _motion_diagnostic_path(path):
    values=[]
    it=_iter_frames(path)
    try: previous=next(it)
    except StopIteration: return {"translation_motion_rms":None,"translation_change_rms":None}
    for current in it:
        m,r=feature_fallback(previous,current,3.0)
        if m is not None:
            values.append([m[0,2],m[1,2],np.arctan2(m[1,0],m[0,0]),r])
        previous=current
    x=np.asarray(values)
    if len(x)==0: return {"translation_motion_rms":None,"translation_change_rms":None}
    return {
      "translation_motion_rms":float(np.sqrt(np.mean(np.sum(x[:,:2]**2,axis=1)))),
      "translation_change_rms":float(np.sqrt(np.mean(np.sum(np.diff(x[:,:2],axis=0)**2,axis=1)))) if len(x)>1 else 0.,
      "rotation_motion_rms":float(np.sqrt(np.mean(x[:,2]**2))),
      "mean_tracking_inlier_ratio":float(x[:,3].mean())}


def stabilize(input_path, output_path, *, backend="raft", device="auto", working_width=480,
              sigma=3.3, radius=10, grid=(18,32), per_cell=2,
              strengths=(1.,1.,0.,0.), min_crop=.70):
    input_path, output_path = Path(input_path), Path(output_path)
    fps,w,h,declared_count=_video_info(input_path)
    frame_iter=_iter_frames(input_path)
    try: previous=next(frame_iter)
    except StopIteration: raise ValueError("Video must contain at least three frames")
    est=FlowEstimator(backend,device,working_width); pairwise=[]; rows=[]
    for i,current in enumerate(frame_iter):
        f,r,scale=est.bidirectional(previous,current)
        mask,err=consistency(f,r)
        src,dst=grid_sample(f,mask,err,*grid,per_cell)
        m,inlier,_=estimate_similarity(src,dst,2.5*scale)
        accepted=m is not None and mask.mean()>=.05 and len(src)>=32 and inlier>=.35
        used="dense"
        if not accepted:
            m,fr=feature_fallback(cv2.resize(previous,(f.shape[1],f.shape[0])),cv2.resize(current,(f.shape[1],f.shape[0])))
            inlier=fr; used="feature"
        if m is None: m=np.eye(2,3); used="identity"
        m=m.copy(); m[:,2]/=scale
        pairwise.append(m)
        rows.append({"pair":i,"reliable_fraction":float(mask.mean()),"samples":len(src),
                     "inlier_ratio":inlier,"estimator":used,"tx":float(m[0,2]),"ty":float(m[1,2])})
        total=max(0,declared_count-1)
        print(f"pair {i+1}/{total or '?'}: reliable={mask.mean():.3f}, inliers={inlier:.3f}, {used}")
        previous=current
    frame_count=len(pairwise)+1
    if frame_count<3: raise ValueError("Video must contain at least three frames")
    poses=accumulate(pairwise); target,raw_p,smooth_p=smooth_poses(poses,sigma,radius)
    warps=compensation(poses,target,strengths); crop=safe_center_crop(warps,w,h,min_crop)
    x0,y0=int(w*(1-crop)/2),int(h*(1-crop)/2); x1,y1=w-x0,h-y0
    with tempfile.TemporaryDirectory() as td:
        silent=Path(td)/"silent.mp4"
        writer=cv2.VideoWriter(str(silent),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
        rendered=0
        for frame,W in zip(_iter_frames(input_path),warps):
            z=cv2.warpAffine(frame,W[:2],(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            z=cv2.resize(z[y0:y1,x0:x1],(w,h),interpolation=cv2.INTER_LINEAR)
            writer.write(z); rendered+=1
        writer.release()
        if rendered != frame_count:
            raise RuntimeError(f"Rendered {rendered} frames, expected {frame_count}")
        cmd=["ffmpeg","-y","-loglevel","error","-i",str(silent),"-i",str(input_path),
             "-map","0:v:0","-map","1:a?","-c:v","libx264","-preset","medium","-crf","18","-c:a","copy",str(output_path)]
        subprocess.run(cmd,check=True)
    before,after=jitter_rms(raw_p),jitter_rms(smooth_p)
    metrics={"input":str(input_path),"output":str(output_path),"backend":backend,"frames":frame_count,
      "width":w,"height":h,"fps":fps,"mean_reliable_fraction":float(np.mean([x["reliable_fraction"] for x in rows])),
      "mean_ransac_inlier_ratio":float(np.mean([x["inlier_ratio"] for x in rows])),
      "feature_fallbacks":sum(x["estimator"]=="feature" for x in rows),"crop_scale":crop,"zoom":1/crop,
      "modeled_jitter_before":before,"modeled_jitter_after":after,
      "modeled_jitter_reduction_percent":100*(1-after/before) if before else 0,
      "independent_input":_motion_diagnostic_path(input_path),"independent_output":_motion_diagnostic_path(output_path)}
    stem=output_path.with_suffix("")
    stem.with_name(stem.name+"_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf8")
    with stem.with_name(stem.name+"_pairs.csv").open("w",newline="") as f:
        cw=csv.DictWriter(f,fieldnames=rows[0]); cw.writeheader(); cw.writerows(rows)
    np.savetxt(stem.with_name(stem.name+"_trajectory.csv"),np.c_[np.arange(len(raw_p)),raw_p,smooth_p],delimiter=",",
               header="frame,raw_tx,raw_ty,raw_theta,raw_log_scale,smooth_tx,smooth_ty,smooth_theta,smooth_log_scale",comments="")
    return metrics
