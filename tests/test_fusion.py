import numpy as np

from vstab.fusion import (adaptive_innovation_fusion,
                          gyro_frequency_split_fusion,
                          robust_sensor_calibration)


def test_robust_sensor_calibration_rejects_outlier():
    rng = np.random.default_rng(4)
    sensor = rng.normal(size=(300, 3))
    truth = sensor @ np.array([[2., 0., 0.], [0., -3., 0.], [0., 0., .5]])
    visual = truth + rng.normal(scale=.02, size=truth.shape)
    visual[120] += 30.
    pred, _, _, weights = robust_sensor_calibration(sensor, visual)
    assert np.median(np.linalg.norm(pred - truth, axis=1)) < .05
    assert weights[120] < .1


def test_adaptive_fusion_trusts_sensor_when_vision_is_weak():
    visual = np.array([[10., 0., 0.], [10., 0., 0.]])
    sensor = np.zeros_like(visual)
    fused, diag = adaptive_innovation_fusion(
        visual, sensor, np.array([.95, .05]), np.ones(3))
    assert diag["sensor_weight"][1] > diag["sensor_weight"][0]
    assert abs(fused[1, 0]) < abs(fused[0, 0])


def test_frequency_split_preserves_fast_sensor_motion():
    n = 180
    t = np.arange(n)
    visual = np.column_stack((.02 * t, np.zeros(n), np.zeros(n)))
    sensor = visual.copy()
    sensor[:, 0] += np.where(t % 2, 1., -1.)
    fused, _ = gyro_frequency_split_fusion(visual, sensor, np.ones(n), 30., .5)
    assert np.std(fused[:, 0]) > np.std(visual[:, 0])
