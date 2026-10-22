#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from vstab.fusion import (adaptive_innovation_fusion, fixed_weight_fusion,
                          gyro_frequency_split_fusion, path_jitter,
                          robust_sensor_calibration)
from vstab.motion import (accumulate, compensation, matrix_from_params,
                          params_from_matrix, safe_center_crop, smooth_poses)


def iter_frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def estimate_pairwise(path: Path, work_width=480, half=None):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    values, confidences, ratios, counts = [], [], [], []

    def prepare(frame):
        if half == "left":
            frame = frame[:, :frame.shape[1] // 2]
        elif half == "right":
            frame = frame[:, frame.shape[1] // 2:]
        scale = work_width / frame.shape[1]
        small = cv2.resize(frame, (work_width, max(2, round(frame.shape[0] * scale))))
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), scale

    ok, first = cap.read()
    if not ok:
        cap.release()
        raise ValueError(f"Cannot read {path}")
    previous, scale = prepare(first)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        current, _ = prepare(frame)
        points = cv2.goodFeaturesToTrack(previous, 900, .01, 7, blockSize=5)
        matrix = None
        ratio = 0.
        good_count = 0
        if points is not None:
            forward, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None,
                                                          winSize=(21, 21), maxLevel=3)
            backward, back_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, forward, None,
                                                                 winSize=(21, 21), maxLevel=3)
            fb = np.linalg.norm(points.reshape(-1, 2) - backward.reshape(-1, 2), axis=1)
            good = status.ravel().astype(bool) & back_status.ravel().astype(bool) & (fb < 1.5)
            src = points.reshape(-1, 2)[good]
            dst = forward.reshape(-1, 2)[good]
            good_count = len(src)
            if good_count >= 8:
                matrix, inliers = cv2.estimateAffinePartial2D(
                    src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0,
                    maxIters=2000, confidence=.995, refineIters=10)
                if inliers is not None:
                    ratio = float(inliers.mean())
        if matrix is None:
            matrix = np.eye(2, 3)
        matrix = matrix.copy()
        matrix[:, 2] /= scale
        param = params_from_matrix(np.vstack((matrix, [0, 0, 1])))[:3]
        density = min(1., good_count / 160.)
        confidence = float(np.clip(ratio * density, .02, 1.))
        values.append(param)
        confidences.append(confidence)
        ratios.append(ratio)
        counts.append(good_count)
        previous = current
    cap.release()
    return np.asarray(values), np.asarray(confidences), {
        "fps": fps, "width": width, "height": height,
        "mean_inlier_ratio": float(np.mean(ratios)),
        "mean_tracks": float(np.mean(counts)),
    }


def load_sensor_delta(frames_csv: Path, pair_count: int):
    with frames_csv.open(encoding="utf8") as stream:
        rows = list(csv.DictReader(stream))
    # Row i records the IMU integral ending at video frame i. Pair 0->1 uses row 1.
    rows = rows[1:pair_count + 1]
    values = [[float(r["delta_x_rad"]), float(r["delta_y_rad"]),
               float(r["delta_z_rad"])] for r in rows]
    valid = np.array([r["gyro_valid"].lower() == "true" for r in rows])
    return np.asarray(values), valid


def method_metrics(params, fps, width, height):
    mats = [matrix_from_params([p[0], p[1], p[2], 0.])[:2] for p in params]
    poses = accumulate(mats)
    sigma = max(2., fps * .65)
    target, raw, smooth = smooth_poses(poses, sigma=sigma, radius=int(round(3 * sigma)))
    warps = compensation(poses, target, strengths=(1., 1., 1., 0.))
    crop = safe_center_crop(warps, width, height, minimum=.70)
    return {
        "increment_jitter": path_jitter(params),
        "smoothed_path_jitter": path_jitter(np.diff(smooth[:, :3], axis=0)),
        "crop_scale": float(crop), "zoom": float(1. / crop),
        "poses": poses, "warps": warps,
    }


def motion_summary(params, confidence):
    if not len(params):
        return {}
    translation = np.linalg.norm(params[:, :2], axis=1)
    return {
        "translation_rms": float(np.sqrt(np.mean(translation ** 2))),
        "translation_change_rms": float(np.sqrt(np.mean(np.diff(params[:, :2], axis=0) ** 2))),
        "rotation_rms": float(np.sqrt(np.mean(params[:, 2] ** 2))),
        "mean_confidence": float(np.mean(confidence)),
    }


def render(path: Path, output: Path, warps, crop):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    x0, y0 = int(width * (1 - crop) / 2), int(height * (1 - crop) / 2)
    x1, y1 = width - x0, height - y0
    for frame, warp in zip(iter_frames(path), warps):
        fixed = cv2.warpAffine(frame, warp[:2], (width, height), flags=cv2.INTER_LINEAR)
        fixed = cv2.resize(fixed[y0:y1, x0:x1], (width, height))
        writer.write(fixed)
    writer.release()


def export_gcsv(sensor_csv: Path, video_path: Path, json_path: Path, output: Path):
    metadata = json.loads(json_path.read_text(encoding="utf8"))
    readout = metadata.get("capture", {}).get("rollingShutterSkewNs", 0) / 1e6
    with sensor_csv.open(encoding="utf8") as source:
        rows = list(csv.DictReader(source))
    gyro = [r for r in rows if r["type"] == "4"]
    accel = [r for r in rows if r["type"] == "1"]
    if not gyro:
        return
    t0 = int(gyro[0]["sensor_timestamp_ns"])
    # Nearest accelerometer sample; output floats are valid in GCSV v1.3.
    accel_t = np.array([int(r["sensor_timestamp_ns"]) for r in accel], dtype=np.int64)
    lines = ["GYROFLOW IMU LOG", "version,1.3", "id,xiaomi_mi9_lab_capture",
             "orientation,XYZ", "vendor,Xiaomi", f"videofilename,{video_path.name}",
             f"frame_readout_time,{readout:.6f}", "frame_readout_direction,0",
             "tscale,0.000000001", "gscale,1", "ascale,0.101971621",
             "t,gx,gy,gz,ax,ay,az"]
    for g in gyro:
        ts = int(g["sensor_timestamp_ns"])
        idx = int(np.argmin(np.abs(accel_t - ts))) if len(accel_t) else -1
        a = accel[idx] if idx >= 0 else {"x": "0", "y": "0", "z": "0"}
        lines.append(f"{ts-t0},{g['x']},{g['y']},{g['z']},{a['x']},{a['y']},{a['z']}")
    output.write_text("\n".join(lines) + "\n", encoding="utf8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-stress", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    capture_dirs = sorted(p for p in args.data.iterdir() if p.is_dir())
    stress_name = capture_dirs[-1].name if capture_dirs else ""
    for capture in capture_dirs:
        originals = list(capture.glob("*.original.mp4"))
        if not originals:
            continue
        original = originals[0]
        stem = original.name.replace(".original.mp4", "")
        comparison = capture / f"{stem}.mp4"
        frame_csv = capture / f"{stem}.frames.csv"
        sensor_csv = capture / f"{stem}.sensors.csv"
        json_path = capture / f"{stem}.json"
        print(f"Analyzing {stem}", flush=True)
        visual, confidence, info = estimate_pairwise(original)
        sensor, valid = load_sensor_delta(frame_csv, len(visual))
        n = min(len(visual), len(sensor))
        visual, confidence, sensor, valid = visual[:n], confidence[:n], sensor[:n], valid[:n]
        confidence = confidence * np.where(valid, 1., .15)
        sensor_pred, beta, residual_scale, calibration_weight = robust_sensor_calibration(
            sensor, visual, confidence)
        fixed = fixed_weight_fusion(visual, sensor_pred, .35)
        adaptive, adaptive_diag = adaptive_innovation_fusion(
            visual, sensor_pred, confidence, residual_scale)
        spectral, spectral_diag = gyro_frequency_split_fusion(
            visual, sensor_pred, confidence, info["fps"], .65)
        methods = {"vision": visual, "fixed_035": fixed,
                   "adaptive": adaptive, "gyro_frequency_split": spectral}
        metrics = {}
        for name, values in methods.items():
            detail = method_metrics(values, info["fps"], info["width"], info["height"])
            metrics[name] = {k: v for k, v in detail.items() if k not in ("poses", "warps")}
        left, left_conf, _ = estimate_pairwise(comparison, half="left")
        right, right_conf, _ = estimate_pairwise(comparison, half="right")
        v7_left, v7_right = motion_summary(left, left_conf), motion_summary(right, right_conf)
        v7_reduction = 100 * (1 - v7_right["translation_change_rms"] /
                              max(v7_left["translation_change_rms"], 1e-9))
        record = {
            "capture": stem, "frames": n + 1, **info,
            "calibration_matrix": beta.tolist(),
            "calibration_residual_scale": residual_scale.tolist(),
            "calibration_outlier_fraction": float(np.mean(calibration_weight < .25)),
            "adaptive_mean_sensor_weight": float(np.mean(adaptive_diag["sensor_weight"])),
            "adaptive_p95_innovation": float(np.percentile(adaptive_diag["innovation"], 95)),
            "v7_equal_crop": v7_left, "v7_stabilized": v7_right,
            "v7_translation_change_reduction_percent": float(v7_reduction),
            "methods": metrics,
        }
        summaries.append(record)
        (args.output / f"{stem}_benchmark.json").write_text(
            json.dumps(record, indent=2), encoding="utf8")
        with (args.output / f"{stem}_pairwise.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["frame", "confidence", "sensor_weight", "innovation",
                             "vision_tx", "vision_ty", "vision_theta",
                             "sensor_tx", "sensor_ty", "sensor_theta",
                             "adaptive_tx", "adaptive_ty", "adaptive_theta"])
            for i in range(n):
                writer.writerow([i + 1, confidence[i], adaptive_diag["sensor_weight"][i],
                                 adaptive_diag["innovation"][i], *visual[i], *sensor_pred[i],
                                 *adaptive[i]])
        export_gcsv(sensor_csv, original, json_path, args.output / f"{stem}.gcsv")
        if args.render_stress and capture.name == stress_name:
            for name in ("fixed_035", "adaptive", "gyro_frequency_split"):
                detail = method_metrics(methods[name], info["fps"], info["width"], info["height"])
                render(original, args.output / f"{stem}_{name}.mp4",
                       detail["warps"], detail["crop_scale"])
    (args.output / "mi9_benchmark_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf8")
    rows = []
    for r in summaries:
        base = r["methods"]["fixed_035"]["increment_jitter"]
        for name, m in r["methods"].items():
            rows.append({"capture": r["capture"], "method": name,
                         "increment_jitter": m["increment_jitter"],
                         "vs_fixed_percent": 100 * (1 - m["increment_jitter"] / max(base, 1e-9)),
                         "crop_scale": m["crop_scale"], "zoom": m["zoom"],
                         "v7_translation_change_reduction_percent": r["v7_translation_change_reduction_percent"]})
    with (args.output / "mi9_benchmark_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
