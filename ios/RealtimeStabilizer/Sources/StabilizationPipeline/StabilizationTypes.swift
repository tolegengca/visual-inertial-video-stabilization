import Foundation
import simd

public enum StabilizationMode: String, Sendable, CaseIterable {
    case gyro
    case hybrid
    case visual
}

public struct StabilizationPipelineConfiguration: Sendable {
    public var mode: StabilizationMode
    public var smoothingWindow: Int
    public var cropScale: Float
    public var visualGridRows: Int
    public var visualGridColumns: Int
    public var externalTelephotoEnabled: Bool
    public var externalTelephotoMultiplier: Float

    public init(
        mode: StabilizationMode = .visual,
        smoothingWindow: Int = 15,
        cropScale: Float = 0.92,
        visualGridRows: Int = 12,
        visualGridColumns: Int = 20,
        externalTelephotoEnabled: Bool = false,
        externalTelephotoMultiplier: Float = 1.0
    ) {
        self.mode = mode
        self.smoothingWindow = max(3, smoothingWindow | 1)
        self.cropScale = min(max(cropScale, 0.70), 1.0)
        self.visualGridRows = max(2, visualGridRows)
        self.visualGridColumns = max(2, visualGridColumns)
        self.externalTelephotoEnabled = externalTelephotoEnabled
        self.externalTelephotoMultiplier = min(max(externalTelephotoMultiplier, 1.0), 4.0)
    }
}

public struct StabilizationDiagnostics: Sendable {
    public let mode: StabilizationMode
    public let visualAccepted: Bool
    public let reliableFraction: Float
    public let inlierRatio: Float
    public let zoom: Float
    public let latencyMilliseconds: Double
    public let intrinsicsAvailable: Bool
    public let rollingShutterConfidence: Float
}

public struct SimilarityTransform: Sendable, Equatable {
    public var translation: SIMD2<Float>
    public var rotation: Float
    public var scale: Float

    public static let identity = SimilarityTransform(translation: .zero, rotation: 0, scale: 1)

    public init(translation: SIMD2<Float>, rotation: Float, scale: Float) {
        self.translation = translation
        self.rotation = rotation
        self.scale = scale
    }

    public func applying(to p: SIMD2<Float>) -> SIMD2<Float> {
        let c = cos(rotation), s = sin(rotation)
        return SIMD2(scale * (c * p.x - s * p.y), scale * (s * p.x + c * p.y)) + translation
    }
}

public enum StabilizationPipelineError: Error {
    case unsupportedPlatform
    case opticalFlowUnavailable
    case invalidFrame
}
