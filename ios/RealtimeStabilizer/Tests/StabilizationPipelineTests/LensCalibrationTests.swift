import XCTest
import simd
@testable import StabilizationPipeline

final class LensCalibrationTests: XCTestCase {
    func testTelephotoScalesFocalLengthsOnly() {
        var k = matrix_identity_float3x3
        k[0,0] = 1200; k[1,1] = 1180; k[2,0] = 960; k[2,1] = 540
        let result = LensCalibration.applyingExternalTelephoto(to:k,enabled:true,multiplier:2)
        XCTAssertEqual(result[0,0],2400)
        XCTAssertEqual(result[1,1],2360)
        XCTAssertEqual(result[2,0],960)
        XCTAssertEqual(result[2,1],540)
    }

    func testTelephotoMultiplierIsClamped() {
        var k = matrix_identity_float3x3; k[0,0] = 100
        XCTAssertEqual(LensCalibration.applyingExternalTelephoto(to:k,enabled:true,multiplier:10)[0,0],400)
    }
}
