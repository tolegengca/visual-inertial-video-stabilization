# Architecture

`FlowEstimator` computes bidirectional dense flow on resized frames. Consistency masks reject mismatched forward/backward correspondences. Spatial grid sampling prevents highly textured regions from dominating the similarity fit.

`pipeline.stabilize` fits adjacent-frame motion with RANSAC, records dense/feature/identity estimator selection, accumulates homogeneous transforms, smooths parameterized poses, and derives compensating warps. A fixed center crop is chosen across the video; rendering makes a second pass through the input rather than retaining all image frames in memory. Pair transforms and trajectory metadata still scale with video length.

The renderer uses OpenCV for warping and an intermediate video, then FFmpeg for H.264 encoding and optional source-audio copying. The output parent directory must already exist. A compatible audio codec is required for stream copy.

`fusion.py` operates on aligned numeric increments. Robust regression maps inertial increments to visual similarity increments. The adaptive and frequency-split methods are experimental interfaces, not an implicit stage of the CLI. Avoid evaluating calibration on the same samples used to fit it when reporting generalization.

The `ios/` package is a separate experimental integration surface. Python verification does not validate its Apple-framework behavior.
