import Foundation
import SwiftAgent
import Logging

// Configure logging
LoggingSystem.bootstrap { label in
    var handler = StreamLogHandler.standardOutput(label: label)
    handler.logLevel = .error
    return handler
}

// Main async function
func runCLI() async {
    do {
            print("🚀 Starting SwiftAgent CLI...")
            print("📥 Loading Qwen3-4B model...")
            print("⏳ First time download may take 5-10 minutes (~2GB)...")
            
            // Create MLX provider with correct HuggingFace model ID
            let provider = try await MLXProvider(
                modelId: "mlx-community/Qwen3-4B-4bit-DWQ-053125"  // Direct HuggingFace model
            )
            
            print("✅ Model loaded successfully!")
            
            // Configure agent
            let config = AgentConfiguration(
                model: "mlx-community/Qwen3-4B-4bit-DWQ-053125",
                provider: "mlx",
                systemPrompt: """
                You are a helpful technical assistant with expertise in MLX and machine learning.
                When asked about documentation or code, be specific and provide examples.
                Use the available tools to browse and search for information.
                """,
                maxTurns: 10,
                temperature: 0.7,
                topP: 0.95,
                repetitionPenalty: 1.15
            )
            
            let agent = Agent(
                configuration: config,
                llmProvider: provider
            )
            
            print("🔌 Connecting to HuggingFace MCP server...")
            
            // Connect to HuggingFace MCP server
            try await agent.connect(servers: [
                MCPServerConfig.http(
                    name: "huggingface",
                    url: URL(string: "https://hf.co/mcp")!,
                    streaming: false
                )
            ])
            
            print("✅ Connected to MCP server!")
            print("\n" + String(repeating: "=", count: 80) + "\n")
            
            // Query about HuggingFace models
            let query = """
            Please help me find information about the Qwen/Qwen2.5-3B model on HuggingFace.
            
            I want to know:
            1. What is the model architecture
            2. What are the key features
            3. What is the model size
            
            Please search for this model information.
            """
            
            print("🔍 Query: \(query)")
            print("\n" + String(repeating: "=", count: 80) + "\n")
            
            // Run the agent
            try await agent.run(query) { @Sendable message in
                await MainActor.run {
                    switch message.role {
                    case .user:
                        print("\n👤 USER:\n\(message.content)\n")
                    case .assistant:
                        print("\n🤖 ASSISTANT:\n\(message.content)\n")
                    case .tool:
                        print("\n🔧 TOOL [\(message.toolName ?? "unknown")]:\n\(message.content)\n")
                    case .system:
                        break // Don't print system messages
                    }
                }
            }
            
            print("\n" + String(repeating: "=", count: 80))
            print("✅ Query completed!")
            
            // Disconnect
            await agent.disconnect()
            
    } catch {
        print("\n❌ Error: \(error)")
        exit(1)
    }
}

// Use RunLoop instead of semaphore for async main
Task {
    await runCLI()
    exit(0)
}

RunLoop.main.run()