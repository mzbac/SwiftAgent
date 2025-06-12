# SwiftAgent

A powerful Swift library for building AI agents that integrate with Model Context Protocol (MCP) servers and various LLM providers.

[![Swift](https://img.shields.io/badge/Swift-6.0-orange.svg)](https://swift.org)
[![Platform](https://img.shields.io/badge/platform-iOS%20%7C%20macOS%20%7C%20tvOS%20%7C%20watchOS-lightgrey.svg)](https://developer.apple.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Features

- 🏠 **On-Device Inference**: Run Qwen3 models locally with MLX (Apple Silicon) 
- 🚀 **MLX Server Support**: Connect to mlx_lm.server for OpenAI-compatible API
- 🔧 **MCP Integration**: Connect to Model Context Protocol servers for tool access
- 🤖 **Tool Calling**: Native support for Qwen3 tool calling capabilities
- 🔄 **Streaming Responses**: Real-time streaming of LLM responses
- 📱 **Cross-Platform**: Supports iOS 16+, macOS 14+ (Apple Silicon)
- 🛡️ **Type-Safe**: Leverages Swift's type system for safe API usage
- ⚡ **Async/Await**: Built with modern Swift concurrency

## Installation

### Swift Package Manager

Add SwiftAgent to your `Package.swift` file:

```swift
dependencies: [
    .package(url: "https://github.com/yourusername/SwiftAgent.git", branch: "main")
]
```

Or add it in Xcode:
1. File → Add Package Dependencies
2. Enter the repository URL
3. Select version and add to your project

## Quick Start

```swift
import SwiftAgent

// Create a local MLX provider with Qwen3 model
let provider = try await MLXProvider(
    modelId: "mlx-community/Qwen3-4B-Instruct-4bit"
)

// Create an agent
let agent = Agent(
    configuration: AgentConfiguration(
        model: "mlx-community/Qwen3-4B-Instruct-4bit",
        provider: "mlx",
        systemPrompt: "You are a helpful assistant.",
        maxTurns: 5
    ),
    llmProvider: provider
)

// Connect to MCP servers (optional)
try await agent.connect(servers: [
    MCPServerConfig.http(
        name: "example",
        url: URL(string: "https://example.com/mcp")!
    )
])

// Run the agent
try await agent.run("Hello! What tools do you have available?") { message in
    print("\(message.role): \(message.content)")
}
```

## Supported Providers

### MLX On-Device Models (Apple Silicon)

Currently only supports Qwen3 models for tool calling:

```swift
// Download and run Qwen3 models locally
let provider = try await MLXProvider(
    modelId: "mlx-community/Qwen3-4B-Instruct-4bit"
)

// Supported Qwen3 models:
// - mlx-community/Qwen3-0.6B-4bit
// - mlx-community/Qwen3-1.7B-4bit  
// - mlx-community/Qwen3-4B-Instruct-4bit
// - mlx-community/Qwen3-8B-Instruct-4bit
```

### MLX Server (OpenAI-Compatible API)

Run mlx_lm.server and connect via OpenAI-compatible API:

```bash
# Start the server
mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-4bit --port 8080
```

```swift
// Connect to the server
let provider = OpenAICompatibleProvider.mlxServer(
    model: "mlx-community/Qwen3-4B-Instruct-4bit",
    host: "localhost",
    port: 8080
)
```

## MCP Server Integration

SwiftAgent supports connecting to Model Context Protocol servers for tool access:

```swift
// HTTP/SSE-based MCP server
let httpServer = MCPServerConfig.http(
    name: "my-tools",
    url: URL(string: "https://api.example.com/mcp")!,
    streaming: false  // Set to false to avoid SSE ping warnings
)

// Connect to servers
try await agent.connect(servers: [httpServer])

// The agent will automatically discover and use available tools
```

## Configuration

### Agent Configuration

```swift
let config = AgentConfiguration(
    model: "mlx-community/Qwen3-4B-Instruct-4bit",    // Model identifier
    provider: "mlx",                                   // "mlx" or "mlx-server"
    systemPrompt: "...",                               // System instructions
    maxTurns: 10,                                     // Max conversation turns
    temperature: 0.7,                                 // Sampling temperature (0.0-1.0)
    maxTokens: 2000,                                  // Max tokens per response
    topP: 0.95,                                       // Nucleus sampling (0.0-1.0)
    repetitionPenalty: 1.15,                          // Repetition penalty (1.0+)
    repetitionContextSize: 20                         // Context window for repetition
)
```

### Logging

SwiftAgent uses SwiftLog for structured logging:

```swift
import Logging

// Configure logging
LoggingSystem.bootstrap { label in
    var handler = StreamLogHandler.standardOutput(label: label)
    handler.logLevel = .debug
    return handler
}
```

## Examples

### Deep Wikipedia Search with Qwen

This example shows how to use Qwen3 with MCP tools for deep research:

```swift
import SwiftAgent

// Create Qwen3 provider for advanced reasoning
let provider = try await MLXProvider(
    modelId: "mlx-community/Qwen3-8B-Instruct-4bit"
)

// Configure agent with research-focused prompt
let config = AgentConfiguration(
    model: "mlx-community/Qwen3-8B-Instruct-4bit",
    provider: "mlx",
    systemPrompt: """
    You are a research assistant with access to web browsing capabilities.
    When asked about a topic, you should:
    1. Search for comprehensive information
    2. Read multiple sources to get a complete picture
    3. Synthesize the information into a well-structured response
    4. Cite your sources when possible
    Think step by step and be thorough in your research.
    """,
    maxTurns: 10,
    temperature: 0.7,
    topP: 0.95,
    repetitionPenalty: 1.15
)

let agent = Agent(configuration: config, llmProvider: provider)

// Connect to MCP server with web browsing tools
try await agent.connect(servers: [
    MCPServerConfig.http(
        name: "web-browser",
        url: URL(string: "http://localhost:3000/mcp")!,
        streaming: false
    )
])

// Perform deep research
try await agent.run("""
    Research the history and impact of the Transformer architecture in AI.
    I want to understand:
    - Who invented it and when
    - Key innovations it introduced  
    - How it revolutionized NLP and AI
    - Major models built on this architecture
    Please provide a comprehensive overview.
""") { message in
    print("\n[\(message.role.rawValue.uppercased())]")
    print(message.content)
}
```

### Configuring Generation Parameters

```swift
// Fine-tune generation for different use cases
let config = AgentConfiguration(
    model: "mlx-community/Qwen3-4B-Instruct-4bit",
    provider: "mlx",
    systemPrompt: "You are a creative writing assistant.",
    temperature: 0.9,        // Higher for creativity
    topP: 0.95,             // Nucleus sampling
    repetitionPenalty: 1.2,  // Reduce repetition
    repetitionContextSize: 40 // Look back 40 tokens
)
```

## API Reference

### Core Types

- `Agent` - Main agent class for managing conversations
- `AgentConfiguration` - Configuration for agent behavior
- `AgentMessage` - Message structure for conversations
- `MCPServerConfig` - MCP server connection configuration
- `LLMProvider` - Protocol for LLM providers

### Key Methods

```swift
// Connect to MCP servers
func connect(servers: [MCPServerConfig]) async throws

// Run agent with user input
func run(_ input: String, onMessage: @escaping (AgentMessage) async -> Void) async throws

// Get conversation history
func getMessages() -> [AgentMessage]

// Clear conversation (except system prompt)
func clearHistory()

// Disconnect from MCP servers
func disconnect() async
```

## License

SwiftAgent is available under the MIT license. See the [LICENSE](LICENSE) file for more info.

## Acknowledgments

- Built on top of the [Model Context Protocol Swift SDK](https://github.com/modelcontextprotocol/swift-sdk)
- Inspired by the MCP ecosystem and agent architectures