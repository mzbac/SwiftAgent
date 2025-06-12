import Foundation

/// Protocol for LLM providers
public protocol LLMProvider: Sendable {
    func complete(
        messages: [AgentMessage],
        tools: [ChatCompletionInputTool],
        temperature: Double,
        maxTokens: Int?,
        topP: Double?,
        repetitionPenalty: Double?,
        repetitionContextSize: Int?
    ) async throws -> AsyncThrowingStream<ChatCompletionStreamOutput, Swift.Error>
}

/// Errors that can occur with LLM providers
public enum LLMProviderError: LocalizedError {
    case invalidResponse
    case streamingError(String)
    case networkError(Swift.Error)
    case authenticationError
    case rateLimitExceeded
    case serverError(Int, String?)
    
    public var errorDescription: String? {
        switch self {
        case .invalidResponse:
            return "Invalid response from LLM provider"
        case .streamingError(let message):
            return "Streaming error: \(message)"
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        case .authenticationError:
            return "Authentication failed. Please check your API key."
        case .rateLimitExceeded:
            return "Rate limit exceeded. Please try again later."
        case .serverError(let code, let message):
            return "Server error (\(code)): \(message ?? "Unknown error")"
        }
    }
}