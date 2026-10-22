#if canImport(AVFoundation) && canImport(CoreImage) && canImport(CoreVideo)
import AVFoundation
import CoreImage
import CoreVideo
import QuartzCore
import simd

public struct StabilizationFrameResult {
    public let image: CIImage
    public let diagnostics: StabilizationDiagnostics
    public let effectiveIntrinsics: simd_float3x3?
}

public final class StabilizationPipeline: @unchecked Sendable {
    private var configuration: StabilizationPipelineConfiguration
    private let visual = VisualMotionEstimator()
    private let motion = MotionStabilizer()
    private var previous: CVPixelBuffer?
    private var previousTimestamp: CMTime?
    private var trajectory: [SimilarityTransform] = [.identity]

    public init(configuration: StabilizationPipelineConfiguration = .init()) {
        self.configuration = configuration
        if configuration.mode != .visual { motion.start() }
    }

    deinit { motion.stop() }

    public func updateConfiguration(_ newValue: StabilizationPipelineConfiguration) {
        let calibrationChanged = configuration.externalTelephotoEnabled != newValue.externalTelephotoEnabled ||
            configuration.externalTelephotoMultiplier != newValue.externalTelephotoMultiplier
        configuration = newValue
        if newValue.mode == .visual { motion.stop() } else { motion.start() }
        if calibrationChanged { reset() }
    }

    public func reset() {
        previous = nil; previousTimestamp = nil; trajectory = [.identity]
    }

    public func processFrame(pixelBuffer: CVPixelBuffer, timestamp: CMTime, intrinsics: simd_float3x3?) throws -> StabilizationFrameResult {
        let started = CACurrentMediaTime()
        let calibrated = intrinsics.map { LensCalibration.applyingExternalTelephoto(to:$0,
            enabled:configuration.externalTelephotoEnabled,multiplier:configuration.externalTelephotoMultiplier) }
        var estimate: SimilarityTransform = .identity
        var reliable: Float = 0, inlier: Float = 0, rsConfidence: Float = 0, accepted = false
        if let previous {
            if configuration.mode == .visual || configuration.mode == .hybrid,
               let result = try visual.estimate(previous:previous,current:pixelBuffer,
                    rows:configuration.visualGridRows,columns:configuration.visualGridColumns) {
                estimate = result.transform; reliable = result.reliableFraction; inlier = result.inlierRatio
                rsConfidence = result.rollingShutterConfidence
                accepted = reliable >= 0.05 && inlier >= 0.35
            }
            if configuration.mode == .gyro, let t0 = previousTimestamp,
               let roll = motion.rollDelta(from:t0.seconds,to:timestamp.seconds) {
                estimate = SimilarityTransform(translation:.zero,rotation:roll,scale:1); accepted = true
            }
        }
        trajectory.append(estimate)
        if trajectory.count > configuration.smoothingWindow { trajectory.removeFirst() }
        let correction = smoothedCorrection()
        let input = CIImage(cvPixelBuffer: pixelBuffer)
        let center = SIMD2(Float(input.extent.midX),Float(input.extent.midY))
        let t = CGAffineTransform(translationX:CGFloat(center.x+correction.translation.x),y:CGFloat(center.y+correction.translation.y))
            .rotated(by:CGFloat(correction.rotation)).scaledBy(x:CGFloat(correction.scale),y:CGFloat(correction.scale))
            .translatedBy(x:CGFloat(-center.x),y:CGFloat(-center.y))
        let warped = input.transformed(by:t)
        let scale = CGFloat(configuration.cropScale)
        let crop = CGRect(x:input.extent.midX-input.extent.width*scale/2,y:input.extent.midY-input.extent.height*scale/2,
                          width:input.extent.width*scale,height:input.extent.height*scale)
        let output = warped.cropped(to:crop).transformed(by:CGAffineTransform(scaleX:1/scale,y:1/scale))
        self.previous = pixelBuffer; previousTimestamp = timestamp
        let diagnostics = StabilizationDiagnostics(mode:configuration.mode,visualAccepted:accepted,
            reliableFraction:reliable,inlierRatio:inlier,zoom:1/configuration.cropScale,
            latencyMilliseconds:(CACurrentMediaTime()-started)*1000,intrinsicsAvailable:intrinsics != nil,
            rollingShutterConfidence:rsConfidence)
        return StabilizationFrameResult(image:output,diagnostics:diagnostics,effectiveIntrinsics:calibrated)
    }

    private func smoothedCorrection() -> SimilarityTransform {
        guard trajectory.count > 1 else { return .identity }
        let n = Float(trajectory.count)
        let meanT = trajectory.reduce(SIMD2<Float>.zero) { $0+$1.translation }/n
        let meanR = trajectory.reduce(Float(0)) { $0+$1.rotation }/n
        let meanS = trajectory.reduce(Float(0)) { $0+$1.scale }/n
        let latest = trajectory.last!
        return SimilarityTransform(translation:meanT-latest.translation,rotation:meanR-latest.rotation,
                                   scale:latest.scale == 0 ? 1 : meanS/latest.scale)
    }
}
#endif
