from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def _mad_scale(x: np.ndarray, axis=0) -> np.ndarray:
    med = np.nanmedian(x, axis=axis)
    mad = np.nanmedian(np.abs(x - med), axis=axis)
    return np.maximum(1.4826 * mad, 1e-6)


def robust_sensor_calibration(sensor_delta, visual_delta, confidence=None,
                              ridge=1e-5, iterations=12):
    """Fit an affine gyro-to-image increment map using IRLS/Huber weights.

    Inputs and outputs are N x 3 arrays: sensor angular increments and visual
    similarity increments [tx, ty, theta].  The intercept absorbs residual bias.
    """
    x = np.asarray(sensor_delta, dtype=float)
    y = np.asarray(visual_delta, dtype=float)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("sensor_delta and visual_delta must both be N x 3")
    base = np.ones(len(x)) if confidence is None else np.clip(confidence, .02, 1.)
    design = np.column_stack((x, np.ones(len(x))))
    weights = base.copy()
    beta = np.zeros((4, 3), dtype=float)
    for _ in range(iterations):
        sw = np.sqrt(weights)[:, None]
        lhs = (design * sw).T @ (design * sw) + ridge * np.eye(4)
        rhs = (design * sw).T @ (y * sw)
        beta = np.linalg.solve(lhs, rhs)
        residual = y - design @ beta
        scale = _mad_scale(residual)
        distance = np.sqrt(np.sum((residual / scale) ** 2, axis=1))
        huber = np.where(distance <= 1.5, 1., 1.5 / np.maximum(distance, 1e-9))
        weights = base * huber
    predicted = design @ beta
    residual_scale = _mad_scale(y - predicted)
    return predicted, beta, residual_scale, weights


def fixed_weight_fusion(visual, sensor_prediction, sensor_weight=.35):
    visual = np.asarray(visual, dtype=float)
    sensor_prediction = np.asarray(sensor_prediction, dtype=float)
    return (1. - sensor_weight) * visual + sensor_weight * sensor_prediction


def adaptive_innovation_fusion(visual, sensor_prediction, confidence,
                               residual_scale, minimum_sensor_weight=.08):
    """Confidence and innovation gated complementary fusion.

    Sensor influence rises when visual tracking is weak, but a large visual/
    inertial innovation suppresses the sensor only when vision itself is good.
    """
    visual = np.asarray(visual, dtype=float)
    sensor_prediction = np.asarray(sensor_prediction, dtype=float)
    confidence = np.clip(np.asarray(confidence, dtype=float), 0., 1.)
    scale = np.maximum(np.asarray(residual_scale, dtype=float), [1., 1., .002])
    innovation = np.sqrt(np.sum(((visual - sensor_prediction) / scale) ** 2, axis=1))
    agreement = np.exp(-.5 * np.minimum(innovation, 8.) ** 2 / 9.)
    sensor_weight = minimum_sensor_weight + (1. - confidence) * .82
    sensor_weight += confidence * .25 * agreement
    sensor_weight = np.clip(sensor_weight, minimum_sensor_weight, .92)
    fused = visual * (1. - sensor_weight[:, None]) + sensor_prediction * sensor_weight[:, None]
    diagnostics = {"sensor_weight": sensor_weight, "innovation": innovation,
                   "agreement": agreement}
    return fused, diagnostics


def gyro_frequency_split_fusion(visual, sensor_prediction, confidence,
                                fps, lowpass_seconds=.65):
    """Fuse low-frequency visual pose with high-frequency inertial motion.

    This follows Gyroflow's separation of responsibilities: vision anchors the
    slow path and IMU carries fast camera motion. Low-confidence visual samples
    are first replaced by the calibrated sensor prediction before integration.
    """
    visual = np.asarray(visual, dtype=float)
    sensor_prediction = np.asarray(sensor_prediction, dtype=float)
    confidence = np.clip(np.asarray(confidence, dtype=float), 0., 1.)
    guarded = visual * confidence[:, None] + sensor_prediction * (1. - confidence[:, None])
    visual_path = np.vstack((np.zeros(3), np.cumsum(guarded, axis=0)))
    sensor_path = np.vstack((np.zeros(3), np.cumsum(sensor_prediction, axis=0)))
    sigma = max(1., float(fps) * lowpass_seconds)
    slow_visual = gaussian_filter1d(visual_path, sigma=sigma, axis=0, mode="reflect")
    slow_sensor = gaussian_filter1d(sensor_path, sigma=sigma, axis=0, mode="reflect")
    fused_path = slow_visual + (sensor_path - slow_sensor)
    return np.diff(fused_path, axis=0), {
        "sigma_frames": sigma, "visual_path": visual_path,
        "sensor_path": sensor_path, "fused_path": fused_path,
    }


def path_jitter(pairwise_params):
    path = np.vstack((np.zeros(3), np.cumsum(np.asarray(pairwise_params), axis=0)))
    if len(path) < 3:
        return 0.0
    accel = np.diff(path, n=2, axis=0)
    return float(np.sqrt(np.mean(np.sum(accel * accel, axis=1))))
