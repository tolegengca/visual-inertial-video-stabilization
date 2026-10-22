import Foundation
import simd

struct VisualMatch: Sendable {
    let source: SIMD2<Float>
    let target: SIMD2<Float>
    let consistencyError: Float
}

enum VisualGeometry {
    static func fitSimilarity(_ matches: [VisualMatch]) -> SimilarityTransform? {
        guard matches.count >= 2 else { return nil }
        let n = Float(matches.count)
        let cs = matches.reduce(SIMD2<Float>.zero) { $0 + $1.source } / n
        let ct = matches.reduce(SIMD2<Float>.zero) { $0 + $1.target } / n
        var a: Float = 0, b: Float = 0, d: Float = 0
        for m in matches {
            let p = m.source - cs, q = m.target - ct
            a += simd_dot(p, q)
            b += p.x*q.y - p.y*q.x
            d += simd_dot(p, p)
        }
        guard d > 1e-6 else { return nil }
        let scale = hypot(a, b) / d
        let angle = atan2(b, a)
        let base = SimilarityTransform(translation: .zero, rotation: angle, scale: scale)
        return SimilarityTransform(translation: ct - base.applying(to: cs), rotation: angle, scale: scale)
    }

    static func ransac(_ matches: [VisualMatch], threshold: Float = 2.5, iterations: Int = 300) -> (SimilarityTransform, Float)? {
        guard matches.count >= 8 else { return nil }
        var state: UInt64 = 0x9e3779b97f4a7c15
        func next(_ upper: Int) -> Int {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Int((state >> 32) % UInt64(upper))
        }
        var best: [VisualMatch] = []
        for _ in 0..<iterations {
            let i = next(matches.count), j = next(matches.count)
            if i == j { continue }
            guard let model = fitSimilarity([matches[i], matches[j]]) else { continue }
            let inliers = matches.filter { simd_length(model.applying(to: $0.source) - $0.target) <= threshold }
            if inliers.count > best.count { best = inliers }
        }
        guard let refined = fitSimilarity(best), best.count >= 8 else { return nil }
        return (refined, Float(best.count) / Float(matches.count))
    }

    static func rollingShutterSlope(_ matches: [VisualMatch], model: SimilarityTransform, height: Float, bands: Int = 6) -> (SIMD2<Float>, Float) {
        var centers: [Float] = [], values: [SIMD2<Float>] = []
        for band in 0..<bands {
            let y0 = Float(band) * height / Float(bands), y1 = Float(band+1) * height / Float(bands)
            let residuals = matches.filter { $0.source.y >= y0 && $0.source.y < y1 }
                .map { $0.target - model.applying(to: $0.source) }
            guard residuals.count >= 6 else { continue }
            centers.append(((y0+y1)*0.5/height)-0.5)
            values.append(residuals.reduce(.zero,+) / Float(residuals.count))
        }
        guard centers.count >= 3 else { return (.zero, 0) }
        let meanX = centers.reduce(0,+)/Float(centers.count)
        let meanY = values.reduce(.zero,+)/Float(values.count)
        let denom = zip(centers,values).reduce(Float(0)) { $0 + pow($1.0-meanX,2) }
        guard denom > 1e-6 else { return (.zero, 0) }
        let slope = zip(centers,values).reduce(SIMD2<Float>.zero) { $0 + ($1.0-meanX)*($1.1-meanY) } / denom
        let confidence = min(1, Float(centers.count)/Float(bands))
        return (slope, confidence)
    }
}
