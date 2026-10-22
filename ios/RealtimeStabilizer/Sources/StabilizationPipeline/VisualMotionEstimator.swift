#if canImport(Vision) && canImport(CoreVideo)
import CoreVideo
import Vision

final class VisualMotionEstimator {
    struct Estimate {
        let transform: SimilarityTransform
        let reliableFraction: Float
        let inlierRatio: Float
        let rollingShutterSlope: SIMD2<Float>
        let rollingShutterConfidence: Float
    }

    private func flow(from source: CVPixelBuffer, to target: CVPixelBuffer) throws -> CVPixelBuffer {
        let request = VNGenerateOpticalFlowRequest(targetedCVPixelBuffer: target, options: [:])
        request.computationAccuracy = .medium
        request.outputPixelFormat = kCVPixelFormatType_TwoComponent32Float
        let handler = VNImageRequestHandler(cvPixelBuffer: source, options: [:])
        try handler.perform([request])
        guard let observation = request.results?.first as? VNPixelBufferObservation else {
            throw StabilizationPipelineError.opticalFlowUnavailable
        }
        return observation.pixelBuffer
    }

    private func vector(_ buffer: CVPixelBuffer, x: Int, y: Int) -> SIMD2<Float> {
        let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: Float.self)
        let stride = CVPixelBufferGetBytesPerRow(buffer) / MemoryLayout<Float>.size
        return SIMD2(base[y*stride + 2*x], base[y*stride + 2*x+1])
    }

    func estimate(previous: CVPixelBuffer, current: CVPixelBuffer, rows: Int, columns: Int) throws -> Estimate? {
        let forward = try flow(from: previous, to: current)
        let backward = try flow(from: current, to: previous)
        CVPixelBufferLockBaseAddress(forward, .readOnly); defer { CVPixelBufferUnlockBaseAddress(forward, .readOnly) }
        CVPixelBufferLockBaseAddress(backward, .readOnly); defer { CVPixelBufferUnlockBaseAddress(backward, .readOnly) }
        let width = CVPixelBufferGetWidth(forward), height = CVPixelBufferGetHeight(forward)
        var accepted = 0, attempted = 0, matches: [VisualMatch] = []
        for r in 0..<rows { for c in 0..<columns {
            let x = min(width-1, (2*c+1)*width/(2*columns)), y = min(height-1, (2*r+1)*height/(2*rows))
            attempted += 1
            let f = vector(forward,x:x,y:y), q = SIMD2(Float(x),Float(y))+f
            let qx = Int(q.x.rounded()), qy = Int(q.y.rounded())
            guard qx >= 0, qx < width, qy >= 0, qy < height else { continue }
            let b = vector(backward,x:qx,y:qy), error = simd_length(f+b)
            guard error <= 1.0 + 0.05*(simd_length(f)+simd_length(b)) else { continue }
            accepted += 1
            matches.append(VisualMatch(source: SIMD2(Float(x),Float(y)), target: q, consistencyError: error))
        }}
        guard let (model,inlier) = VisualGeometry.ransac(matches), attempted > 0 else { return nil }
        let (slope,confidence) = VisualGeometry.rollingShutterSlope(matches,model:model,height:Float(height))
        return Estimate(transform:model,reliableFraction:Float(accepted)/Float(attempted),inlierRatio:inlier,
                        rollingShutterSlope:slope,rollingShutterConfidence:confidence)
    }
}
#endif
