import simd

public enum LensCalibration {
    /// Scales focal lengths for a centered afocal converter. The principal
    /// point and the homogeneous row remain unchanged.
    public static func applyingExternalTelephoto(
        to intrinsics: simd_float3x3,
        enabled: Bool,
        multiplier: Float
    ) -> simd_float3x3 {
        guard enabled else { return intrinsics }
        let m = min(max(multiplier, 1.0), 4.0)
        var calibrated = intrinsics
        calibrated[0, 0] *= m
        calibrated[1, 1] *= m
        return calibrated
    }
}
