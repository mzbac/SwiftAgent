// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "SwiftAgent",
    platforms: [
        .iOS(.v16),
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "SwiftAgent",
            targets: ["SwiftAgent"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/modelcontextprotocol/swift-sdk.git", from: "0.9.0"),
        .package(url: "https://github.com/apple/swift-log.git", from: "1.5.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.25.4"),
        .package(url: "https://github.com/mzbac/mlx-swift-examples.git", revision: "9b5794e6ed994bf078f4793ad27c04d88355d8d9"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.21")
    ],
    targets: [
        .target(
            name: "SwiftAgent",
            dependencies: [
                .product(name: "MCP", package: "swift-sdk"),
                .product(name: "Logging", package: "swift-log"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-examples"),
                .product(name: "MLXLLM", package: "mlx-swift-examples"),
                .product(name: "Transformers", package: "swift-transformers")
            ]
        ),
        .testTarget(
            name: "SwiftAgentTests",
            dependencies: ["SwiftAgent"]
        )
    ]
)