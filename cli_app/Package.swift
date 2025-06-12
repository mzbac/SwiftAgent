// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SwiftAgentCLI",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        .package(name: "SwiftAgent", path: "../")
    ],
    targets: [
        .executableTarget(
            name: "SwiftAgentCLI",
            dependencies: [
                .product(name: "SwiftAgent", package: "SwiftAgent")
            ]
        )
    ]
)