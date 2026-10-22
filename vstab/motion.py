from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


def consistency(flow_f, flow_b, abs_thr=1.0, rel_thr=0.05):
    h, w = flow_f.shape[:2]
    y, x = np.mgrid[:h, :w].astype(np.float32)
    qx, qy = x + flow_f[..., 0], y + flow_f[..., 1]
    valid = (qx >= 0) & (qx <= w - 1) & (qy >= 0) & (qy <= h - 1)
    bx = cv2.remap(flow_b[..., 0], qx, qy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    by = cv2.remap(flow_b[..., 1], qx, qy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    err = np.hypot(flow_f[..., 0] + bx, flow_f[..., 1] + by)
    mag = np.linalg.norm(flow_f, axis=2) + np.hypot(bx, by)
    return valid & (err <= abs_thr + rel_thr * mag), err


def grid_sample(flow, mask, error, rows=18, cols=32, per_cell=2):
    h, w = mask.shape
    src, dst = [], []
    for r in range(rows):
        y0, y1 = r * h // rows, (r + 1) * h // rows
        for c in range(cols):
            x0, x1 = c * w // cols, (c + 1) * w // cols
            ys, xs = np.where(mask[y0:y1, x0:x1])
            if len(xs) == 0:
                continue
            ys, xs = ys + y0, xs + x0
            idx = np.argsort(error[ys, xs])[:per_cell]
            p = np.column_stack((xs[idx], ys[idx])).astype(np.float32)
            src.append(p); dst.append(p + flow[ys[idx], xs[idx]])
    if not src:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    return np.vstack(src), np.vstack(dst)


def estimate_similarity(src, dst, threshold=2.5):
    if len(src) < 4:
        return None, 0.0, None
    m, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=threshold,
        maxIters=3000, confidence=.995, refineIters=10)
    ratio = float(inliers.mean()) if inliers is not None else 0.0
    return m, ratio, inliers


def matrix_from_params(p):
    tx, ty, th, ls = p
    s, c, q = np.exp(ls), np.cos(th), np.sin(th)
    return np.array([[s*c, -s*q, tx], [s*q, s*c, ty], [0, 0, 1]], float)


def params_from_matrix(m):
    return np.array([m[0, 2], m[1, 2], np.arctan2(m[1, 0], m[0, 0]),
                     np.log(max(1e-8, np.hypot(m[0, 0], m[1, 0])))])


def accumulate(pairwise):
    poses = [np.eye(3)]
    for m in pairwise:
        h = np.vstack((m, [0, 0, 1])) if m.shape == (2, 3) else m
        poses.append(h @ poses[-1])
    return np.stack(poses)


def smooth_poses(poses, sigma=3.3, radius=10):
    p = np.stack([params_from_matrix(x) for x in poses])
    p[:, 2] = np.unwrap(p[:, 2])
    truncate = radius / max(sigma, 1e-6)
    smooth = gaussian_filter1d(p, sigma=sigma, axis=0, mode="reflect", truncate=truncate)
    return np.stack([matrix_from_params(x) for x in smooth]), p, smooth


def compensation(poses, target, strengths=(1., 1., 0., 0.)):
    raw = np.stack([params_from_matrix(x) for x in poses])
    dst = np.stack([params_from_matrix(x) for x in target])
    strengths = np.asarray(strengths)
    mixed = raw + strengths * (dst - raw)
    selected = np.stack([matrix_from_params(x) for x in mixed])
    return np.stack([s @ np.linalg.inv(c) for s, c in zip(selected, poses)])


def safe_center_crop(warps, width, height, minimum=.70):
    corners = np.array([[0,0,1],[width-1,0,1],[width-1,height-1,1],[0,height-1,1]], float).T
    def ok(scale):
        x0, x1 = width*(1-scale)/2, width*(1+scale)/2
        y0, y1 = height*(1-scale)/2, height*(1+scale)/2
        pts = np.array([[x0,y0,1],[x1,y0,1],[x1,y1,1],[x0,y1,1]], float).T
        for W in warps:
            q = np.linalg.inv(W) @ pts; q /= q[2]
            if q[0].min() < 0 or q[0].max() >= width or q[1].min() < 0 or q[1].max() >= height:
                return False
        return True
    lo, hi = minimum, 1.0
    if not ok(lo): return minimum
    for _ in range(24):
        mid=(lo+hi)/2
        if ok(mid): lo=mid
        else: hi=mid
    return lo


def jitter_rms(path):
    if len(path) < 3: return 0.0
    return float(np.sqrt(np.mean(np.sum(np.diff(path[:, :2], n=2, axis=0)**2, axis=1))))


def visual_row_motion(src, dst, global_matrix, image_height, bands=6, min_points=8):
    """Estimate a linear rolling-shutter residual from visual matches only.

    The global similarity model is removed first.  Residual translation is
    averaged in horizontal readout bands, then regressed against normalized
    row time in [-0.5, 0.5].  The returned slopes describe start-to-end
    residual motion in pixels.  This small model is intended for confidence
    gated correction; it is not a replacement for a full spatial mesh.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if len(src) != len(dst) or len(src) < max(2 * min_points, 4):
        return None
    predicted = src @ np.asarray(global_matrix)[:, :2].T + np.asarray(global_matrix)[:, 2]
    residual = dst - predicted
    rows, means, weights = [], [], []
    for band in range(bands):
        y0, y1 = band * image_height / bands, (band + 1) * image_height / bands
        keep = (src[:, 1] >= y0) & (src[:, 1] < y1)
        if keep.sum() < min_points:
            continue
        rows.append(((y0 + y1) * .5 / image_height) - .5)
        means.append(np.median(residual[keep], axis=0))
        weights.append(float(keep.sum()))
    if len(rows) < 3:
        return None
    x = np.asarray(rows)
    y = np.asarray(means)
    w = np.sqrt(np.asarray(weights))
    design = np.column_stack((np.ones_like(x), x))
    coeff = np.linalg.lstsq(design * w[:, None], y * w[:, None], rcond=None)[0]
    fitted = design @ coeff
    rmse = float(np.sqrt(np.mean(np.sum((y - fitted) ** 2, axis=1))))
    confidence = float(min(1.0, len(rows) / bands) * np.exp(-rmse / 2.0))
    return {"tx_slope": float(coeff[1, 0]), "ty_slope": float(coeff[1, 1]),
            "rmse": rmse, "confidence": confidence, "bands": len(rows)}
