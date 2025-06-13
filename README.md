# SwiftAgent

A powerful Swift library for building AI agents that integrate with Model Context Protocol (MCP) servers and various LLM providers.

[![Swift](https://img.shields.io/badge/Swift-6.0-orange.svg)](https://swift.org)
[![Platform](https://img.shields.io/badge/platform-iOS%20%7C%20macOS%20%7C%20tvOS%20%7C%20watchOS-lightgrey.svg)](https://developer.apple.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Features

- 🏠 **On-Device Inference**: Run Qwen3 models locally with MLX (Apple Silicon)
- ⚡ **Advanced Prompt Caching**: KV cache management for faster multi-turn conversations
- 🚀 **MLX Server Support**: Connect to mlx_lm.server for OpenAI-compatible API
- 🔧 **MCP Integration**: Connect to Model Context Protocol servers for tool access
- 🤖 **Tool Calling**: Native support for Qwen3 tool calling
- 📱 **Cross-Platform**: Supports iOS 16+, macOS 14+ (Apple Silicon)
- ⚡ **Async/Await**: Built with modern Swift concurrency and actors

## Installation

### Swift Package Manager

Add SwiftAgent to your `Package.swift` file:

```swift
dependencies: [
    .package(url: "https://github.com/mzbac/SwiftAgent.git", branch: "main")
]
```

Or add it in Xcode:
1. File → Add Package Dependencies
2. Enter the repository URL
3. Select version and add to your project

## Architecture Overview

SwiftAgent provides a flexible, modular architecture for building AI agents:

- **Agent**: Main orchestrator managing conversations, tool execution, and state
- **LLM Providers**: Pluggable backends for different model inference approaches
  - `MLXProvider`: Local on-device inference with advanced caching
  - `OpenAICompatibleProvider`: API-based inference for mlx_lm.server
- **MCP Integration**: Full support for Model Context Protocol tools
- **Message System**: Type-safe message handling with role-based routing
- **Error Handling**: Comprehensive error types for robust applications

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
        name: "filesystem",
        url: URL(string: "http://localhost:3000/mcp")!
    )
])

// Run the agent
try await agent.run("What files are in the current directory?") { message in
    print("\(message.role): \(message.content)")
}

// Cleanup
defer { 
    Task { await agent.disconnect() }
}
```

## Supported Providers

### MLX On-Device Models (Apple Silicon)

Currently supports Qwen3 models for tool calling. You can use either HuggingFace models directly or mlx-community quantized versions:

```swift
// Direct HuggingFace models (automatically quantized)
let provider = try await MLXProvider(
    modelId: "Qwen/Qwen3-4B-Instruct"
)

// Pre-quantized mlx-community models
let provider = try await MLXProvider(
    modelId: "mlx-community/Qwen3-4B-Instruct-4bit"
)

// Supported Qwen3 models:
// HuggingFace:
// - Qwen/Qwen3-0.5B-Instruct
// - Qwen/Qwen3-1.5B-Instruct
// - Qwen/Qwen3-4B-Instruct
// - Qwen/Qwen3-8B-Instruct
// - Qwen/Qwen3-14B-Instruct
// - Qwen/Qwen3-32B-Instruct
//
// mlx-community (pre-quantized):
// - mlx-community/Qwen3-0.5B-Instruct-4bit
// - mlx-community/Qwen3-1.5B-Instruct-4bit  
// - mlx-community/Qwen3-4B-Instruct-4bit
// - mlx-community/Qwen3-8B-Instruct-4bit
// - mlx-community/Qwen3-14B-Instruct-4bit
// - mlx-community/Qwen3-32B-Instruct-4bit
```

#### MLX Provider Features

- **Prompt Caching**: Sophisticated KV cache management for fast multi-turn conversations
- **Cache Statistics**: Track cache hits, size, and performance metrics
- **Streaming Metrics**: Real-time token/s and generation statistics
- **Memory Efficient**: Automatic cache cleanup and size management
- **Tool Support**: XML-style tool calling format for Qwen3 models

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

## Advanced Features

### Prompt Caching (MLX Provider)

The MLX provider includes advanced prompt caching that dramatically improves performance in multi-turn conversations:

```swift
// Enable caching (default behavior)
let provider = try await MLXProvider(
    modelId: "Qwen/Qwen3-4B-Instruct",
    enableCaching: true  // Default
)

// Cache statistics are logged automatically:
// [MLXProvider] Cache stats - Hit rate: 85.7%, Size: 1.2GB, Saved tokens: 15420
```

Caching provides:
- Speedup for repeated context
- Automatic cache invalidation on context changes
- Memory-efficient storage with size limits
- Per-conversation isolation

### Tool Calling

Qwen3 models support tool calling through XML-style formatting:

```swift
// Tools are automatically discovered from MCP servers
// The agent handles tool call formatting and execution
try await agent.run("Search for information about Swift concurrency") { message in
    switch message.role {
    case .tool:
        print("Tool executed: \(message.content)")
    case .assistant:
        print("Assistant: \(message.content)")
    default:
        break
    }
}
```

### Error Handling

SwiftAgent provides comprehensive error handling:

```swift
do {
    try await agent.run("Your prompt") { message in
        // Handle messages
    }
} catch AgentError.maxTurnsExceeded {
    print("Conversation too long")
} catch AgentError.toolExecutionFailed(let toolName, let error) {
    print("Tool \(toolName) failed: \(error)")
} catch AgentError.cancelled {
    print("Operation cancelled")
}
```

## Configuration

### Agent Configuration

```swift
let config = AgentConfiguration(
    model: "Qwen/Qwen3-4B-Instruct",                 // Model identifier
    provider: "mlx",                                  // "mlx" or "mlx-server"
    systemPrompt: "...",                              // System instructions
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

### File System Agent with Tool Usage

This example shows practical tool usage with file system operations:

```swift
import SwiftAgent

// Create provider with caching for fast interactions
let provider = try await MLXProvider(
    modelId: "mlx-community/Qwen3-4B-Instruct-4bit"
)

// Create agent with file system access
let agent = Agent(
    configuration: AgentConfiguration(
        model: "mlx-community/Qwen3-4B-Instruct-4bit",
        provider: "mlx",
        systemPrompt: "You are a helpful file system assistant.",
        maxTurns: 10
    ),
    llmProvider: provider
)

// Connect to filesystem MCP server
try await agent.connect(servers: [
    MCPServerConfig.stdio(
        name: "filesystem",
        command: "/usr/local/bin/mcp-server-filesystem",
        args: ["-r", NSHomeDirectory()]
    )
])

// Use the agent for file operations
try await agent.run("""
    Please help me organize my Downloads folder:
    1. List all PDF files
    2. Create a 'PDFs' subfolder if it doesn't exist
    3. Move all PDFs to the new folder
    4. Give me a summary of what was moved
""") { message in
    if message.role == .tool {
        print("🔧 Tool: \(message.content)")
    } else if message.role == .assistant {
        print("\n💬 Assistant: \(message.content)")
    }
}
```

### Research Agent with Web Tools

```swift
import SwiftAgent

// Use a larger model for complex research
let provider = try await MLXProvider(
    modelId: "mlx-community/Qwen3-4B-Instruct-4bit"
)

// Configure for research tasks
let config = AgentConfiguration(
    model: "mlx-community/Qwen3-4B-Instruct-4bit",
    provider: "mlx",
    systemPrompt: """
    You are a thorough research assistant. When researching:
    1. Search for authoritative sources
    2. Verify facts from multiple sources
    3. Provide citations
    4. Structure information clearly
    """,
    maxTurns: 15,  // More turns for deep research
    temperature: 0.7,
    topP: 0.95
)

let agent = Agent(configuration: config, llmProvider: provider)

// Connect to web search and browser tools
try await agent.connect(servers: [
    MCPServerConfig.http(
        name: "web-search",
        url: URL(string: "http://localhost:3001/mcp")!,
        streaming: false
    ),
    MCPServerConfig.http(
        name: "browser",
        url: URL(string: "http://localhost:3002/mcp")!,
        streaming: false
    )
])

// Conduct research
try await agent.run("""
    Research the current state of quantum computing in 2024.
    Focus on recent breakthroughs and commercial applications.
""") { message in
    print(message.content)
}
```

### Code Assistant with Git Integration

```swift
import SwiftAgent

// Setup code assistant
let provider = try await MLXProvider(
    modelId: "Qwen/Qwen3-4B-Instruct"
)

let agent = Agent(
    configuration: AgentConfiguration(
        model: "Qwen/Qwen3-4B-Instruct",
        provider: "mlx",
        systemPrompt: "You are an expert Swift developer.",
        maxTurns: 20
    ),
    llmProvider: provider
)

// Connect to filesystem and git servers
try await agent.connect(servers: [
    MCPServerConfig.filesystem(),  // Pre-configured filesystem access
    MCPServerConfig.git()          // Pre-configured git operations
])

// Use for code tasks
try await agent.run("""
    Review the recent commits in this repository and:
    1. Summarize the main changes
    2. Identify any potential issues
    3. Suggest improvements to the code structure
""") { message in
    print(message.content)
}
```

### Performance Optimization

```swift
// Optimize for different scenarios

// Fast, focused responses
let fastConfig = AgentConfiguration(
    model: "mlx-community/Qwen3-0.6B-Instruct-4bit",  // Smaller model
    provider: "mlx",
    systemPrompt: "Be concise and direct.",
    temperature: 0.3,       // Lower for consistency
    maxTokens: 500,         // Limit response length
    topP: 0.9
)

// Creative, detailed responses  
let creativeConfig = AgentConfiguration(
    model: "mlx-community/Qwen3-0.6B-Instruct-4bit",    // Larger model
    provider: "mlx",
    systemPrompt: "You are a creative writing assistant.",
    temperature: 0.9,        // Higher for creativity
    maxTokens: 2000,         // Allow longer responses
    topP: 0.95,             // Wider sampling
    repetitionPenalty: 1.2,  // Reduce repetition
    repetitionContextSize: 40 // Larger context window
)

// Technical accuracy
let technicalConfig = AgentConfiguration(
    model: "Qwen/Qwen3-4B-Instruct",
    provider: "mlx",
    systemPrompt: "You are a technical expert. Be precise and accurate.",
    temperature: 0.1,        // Very low for determinism
    topP: 0.85,             // Narrower sampling
    repetitionPenalty: 1.05  // Minimal penalty
)
```

### Resource Management

```swift
// Use withAgent for automatic cleanup
try await Agent.withAgent(
    configuration: config,
    llmProvider: provider,
    servers: [MCPServerConfig.filesystem()]
) { agent in
    try await agent.run("Your task") { message in
        print(message.content)
    }
    // Agent automatically disconnects when done
}

// Manual management
let agent = Agent(configuration: config, llmProvider: provider)
defer {
    Task { await agent.disconnect() }
}
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