import Foundation

/// Configuration for agent behavior and LLM parameters.
///
/// This structure defines how the agent operates, including model selection,
/// conversation limits, and generation parameters.
///
/// Example with local MLX model:
/// ```swift
/// let config = AgentConfiguration(
///     model: "mlx-community/Qwen3-4B-Instruct-4bit",
///     provider: "mlx",
///     systemPrompt: "You are a helpful coding assistant.",
///     maxTurns: 5,
///     temperature: 0.8
/// )
/// ```
///
/// Example with mlx_lm.server:
/// ```swift
/// let config = AgentConfiguration(
///     model: "mlx-community/Qwen3-4B-Instruct-4bit",
///     provider: "mlx-server",
///     systemPrompt: "You are a helpful assistant."
/// )
/// ```
public struct AgentConfiguration: Sendable, Codable {
    /// The model identifier (e.g., "mlx-community/Qwen3-4B-Instruct-4bit")
    public let model: String
    
    /// The provider name ("mlx" for local MLX models, "mlx-server" for mlx_lm.server)
    public let provider: String
    
    /// The system prompt that defines the agent's behavior
    public let systemPrompt: String
    
    /// Maximum number of conversation turns before stopping
    public let maxTurns: Int
    
    /// Temperature for response generation (0.0 = deterministic, 1.0 = creative)
    public let temperature: Double
    
    /// Optional maximum tokens per response
    public let maxTokens: Int?
    
    /// Optional top-p sampling parameter (nucleus sampling)
    public let topP: Double?
    
    /// Optional repetition penalty for reducing repetitions
    public let repetitionPenalty: Double?
    
    /// Optional context size for repetition penalty
    public let repetitionContextSize: Int?
    
    /// Creates a new agent configuration.
    /// - Parameters:
    ///   - model: The model identifier (must be a Qwen3 model for MLX provider)
    ///   - provider: The provider name ("mlx" or "mlx-server")
    ///   - systemPrompt: System instructions (defaults to helpful assistant)
    ///   - maxTurns: Maximum conversation turns (default: 10)
    ///   - temperature: Generation temperature (default: 0.7)
    ///   - maxTokens: Optional token limit per response
    public init(
        model: String,
        provider: String,
        systemPrompt: String = AgentConfiguration.defaultSystemPrompt,
        maxTurns: Int = 10,
        temperature: Double = 0.7,
        maxTokens: Int? = nil,
        topP: Double? = nil,
        repetitionPenalty: Double? = nil,
        repetitionContextSize: Int? = nil
    ) {
        self.model = model
        self.provider = provider
        self.systemPrompt = systemPrompt
        self.maxTurns = maxTurns
        self.temperature = temperature
        self.maxTokens = maxTokens
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
    }
    
    public static let defaultSystemPrompt = """
    You are an intelligent assistant. Use the available tools to help answer questions 
    and complete tasks. Think step by step and use tools when needed. Be concise and helpful.
    """
}