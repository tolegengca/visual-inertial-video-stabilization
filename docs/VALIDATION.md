# Validation record

Publication preparation: 2026-10-22, macOS arm64, Python 3.12.

| Check | Result | Scope |
|---|---|---|
| Python geometry + fusion tests | 12 passed | Consistency, RANSAC, transforms, smoothing, crop, row residuals and fusion |
| Synthetic Farneback CLI | Passed | 48 frames, 640 × 360, 24 FPS, H.264 render, AAC audio track retained, and diagnostic outputs |
| Optional RAFT backend | Not executed | Requires torch/torchvision and downloaded model weights |
| Real-capture/IMU benchmark | Not executed in publication validation | Dataset-specific driver; private capture data excluded |
| Separate iOS package | Not built in publication validation | Python checks do not validate Apple framework integration |

## Reproduce the smoke test

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/make_test_video.py
mkdir -p results
python stabilize.py data/synthetic_shaky.mp4 results/stabilized.mp4 --backend farneback --working-width 320
```

FFmpeg must be installed separately. The generated fixture includes a moving foreground rectangle and a synthetic audio tone. The example metrics are a single run, not a real-world leaderboard.

See `examples/synthetic-farneback-metrics.json` for the complete numeric output. On this scene, independent feature-tracked translation RMS was 11.245 px/frame before and 0.720 px/frame after; the fixed crop scale was 0.9602. These are scene-specific motion diagnostics, not a claim of perceptual superiority or performance on arbitrary camera footage.

The README GIF is generated from this exact synthetic input/output. Different library versions may change measurements slightly. No private footage, external ground-truth poses, or hardware camera timing was validated.

Known limits include parallax, moving objects, approximate global similarity motion, crop compromises, codec-dependent audio stream copying, and separately calibrated/timestamp-aligned IMU data. A trajectory smoother alone cannot establish image quality.

## Publication environment

- numpy 2.5.4
- scipy 1.18.1
- opencv-python-headless 4.14.0.94
- pytest 9.1.1
