#!/usr/bin/env python3
from pathlib import Path
import subprocess, tempfile
import cv2, numpy as np

out=Path(__file__).resolve().parents[1]/"data"/"synthetic_shaky.mp4"; out.parent.mkdir(exist_ok=True)
w,h,fps,n=640,360,24,48
rng=np.random.default_rng(7)
base=np.full((h+120,w+160,3),(210,220,225),np.uint8)
for _ in range(500):
    x,y=rng.integers(0,base.shape[1]),rng.integers(0,base.shape[0]); c=rng.integers(30,220,3).tolist()
    cv2.circle(base,(x,y),int(rng.integers(2,8)),c,-1)
cv2.putText(base,"SYNTHETIC STABILIZATION TEST",(130,180),cv2.FONT_HERSHEY_SIMPLEX,1.2,(15,35,80),3)
with tempfile.TemporaryDirectory() as td:
    silent=Path(td)/"silent.mp4"; vw=cv2.VideoWriter(str(silent),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
    truth=[]
    for i in range(n):
        smooth=np.array([25*np.sin(i/24),10*np.cos(i/28)])
        shake=np.array([7*np.sin(i*2.31)+rng.normal(0,2),5*np.sin(i*1.77)+rng.normal(0,1.5)])
        p=smooth+shake; truth.append([i,*smooth,*shake,*p])
        x,y=(80+p).astype(int); fr=base[y:y+h,x:x+w].copy()
        cv2.rectangle(fr,(50+3*i,260),(110+3*i,315),(20,90,230),-1)
        vw.write(fr)
    vw.release()
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(silent),"-f","lavfi","-i",f"sine=frequency=440:duration={n/fps}","-c:v","libx264","-pix_fmt","yuv420p","-c:a","aac","-shortest",str(out)],check=True)
np.savetxt(out.with_name("synthetic_ground_truth.csv"),truth,delimiter=",",header="frame,smooth_x,smooth_y,shake_x,shake_y,total_x,total_y",comments="")
print(out)

