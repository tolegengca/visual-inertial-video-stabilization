# StabilizationPipeline integration

Add `ios/RealtimeStabilizer` as a local Swift Package and link the
`StabilizationPipeline` product to the camera target. The package requires iOS
17 or later.

Create one stateful pipeline and deliver frames serially in capture order:

```swift
let pipeline = StabilizationPipeline(configuration: .init(
    mode: .visual,
    externalTelephotoEnabled: true,
    externalTelephotoMultiplier: 2.0
))

let result = try pipeline.processFrame(
    pixelBuffer: imageBuffer,
    timestamp: CMSampleBufferGetPresentationTimeStamp(sampleBuffer),
    intrinsics: cameraIntrinsicMatrix
)
preview.render(result.image)
```

## Host responsibilities

- Configure AVFoundation capture, permissions, recording and preview.
- Set `alwaysDiscardsLateVideoFrames` according to the product latency budget.
- Call the pipeline from one serial queue; reset after discontinuities or a
  lens/calibration change.
- Disable AVFoundation system stabilization while a custom mode is active.
- Record diagnostics when testing a new device or converter lens.

## External telephoto calibration

The configuration multiplier scales `fx` and `fy` while preserving the
principal point. This models a centered afocal converter only. Physical tests
are still required for optical alignment, radial distortion, chromatic
aberration and decentering.

## Modes

- `visual`: Vision optical flow only; no Core Motion sampling is started.
- `gyro`: Core Motion roll correction without optical flow.
- `hybrid`: visual global motion with Core Motion available to the host for
  device-specific fusion and validation.
