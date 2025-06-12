import Foundation

/// Errors that can occur with the agent
public enum AgentError: LocalizedError {
    case alreadyRunning
    case notConnected
    case invalidServerConfig(String)
    case toolExecutionFailed(String)
    case maxTurnsExceeded
    case cancelled
    case mcpConnectionFailed(Error)
    case llmProviderError(Error)
    
    public var errorDescription: String? {
        switch self {
        case .alreadyRunning:
            return "Agent is already running"
        case .notConnected:
            return "Agent is not connected to any MCP servers"
        case .invalidServerConfig(let message):
            return "Invalid server configuration: \(message)"
        case .toolExecutionFailed(let message):
            return "Tool execution failed: \(message)"
        case .maxTurnsExceeded:
            return "Maximum number of conversation turns exceeded"
        case .cancelled:
            return "Agent operation was cancelled"
        case .mcpConnectionFailed(let error):
            return "Failed to connect to MCP server: \(error.localizedDescription)"
        case .llmProviderError(let error):
            return "LLM provider error: \(error.localizedDescription)"
        }
    }
}