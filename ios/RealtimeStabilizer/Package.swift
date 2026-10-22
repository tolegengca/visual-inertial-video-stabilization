// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "StabilizationPipeline",
    platforms: [.iOS(.v17)],
    products: [
        .library(name: "StabilizationPipeline", targets: ["StabilizationPipeline"])
    ],
    targets: [
        .target(name: "StabilizationPipeline"),
        .testTarget(name: "StabilizationPipelineTests", dependencies: ["StabilizationPipeline"])
    ]
)
