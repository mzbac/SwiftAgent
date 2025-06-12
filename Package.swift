// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

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
        ),
    ],
    dependencies: [
        // Model Context Protocol SDK
        .package(url: "https://github.com/modelcontextprotocol/swift-sdk.git", from: "0.9.0"),
        // Swift logging API
        .package(url: "https://github.com/apple/swift-log.git", from: "1.5.0"),
        // MLX Swift for core functionality
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.25.4"),
        // MLX Swift Examples for LLM support
        .package(url: "https://github.com/mzbac/mlx-swift-examples.git", revision: "a7031a6eaa8422c7b73829125799445f5ae565f5"),
        // Transformers for tokenization
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