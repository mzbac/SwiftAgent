import Foundation
import MCP
import Logging

/// A provider that works with OpenAI-compatible APIs, primarily designed for mlx_lm.server
///
/// This provider is intended to be used with mlx_lm.server, which provides an OpenAI-compatible
/// API for running MLX models. While it can technically work with other OpenAI-compatible APIs,
/// it's primarily tested and optimized for mlx_lm.server.
///
/// Example usage with mlx_lm.server:
/// ```swift
/// let provider = OpenAICompatibleProvider.mlxServer(
///     model: "mlx-community/Qwen3-4B-Instruct-4bit",
///     host: "localhost",
///     port: 8080
/// )
/// ```
public struct OpenAICompatibleProvider: LLMProvider {
    private let apiKey: String
    private let baseURL: URL
    private let model: String
    private let session: URLSession
    private let logger: Logger?
    
    public init(
        baseURL: URL,
        apiKey: String,
        model: String,
        session: URLSession = .shared,
        logger: Logger? = nil
    ) {
        self.baseURL = baseURL
        self.apiKey = apiKey
        self.model = model
        self.session = session
        self.logger = logger
    }
    
    public func complete(
        messages: [AgentMessage],
        tools: [ChatCompletionInputTool],
        temperature: Double,
        maxTokens: Int?,
        topP: Double? = nil,
        repetitionPenalty: Double? = nil,
        repetitionContextSize: Int? = nil
    ) async throws -> AsyncThrowingStream<ChatCompletionStreamOutput, Swift.Error> {
        let openAIMessages = messages.map { $0.toOpenAIFormat() }
        
        var body: [String: Any] = [
            "model": model,
            "messages": openAIMessages,
            "temperature": temperature,
            "stream": true
        ]
        
        if let maxTokens = maxTokens {
            body["max_tokens"] = maxTokens
        }
        
        if let topP = topP {
            body["top_p"] = topP
        }
        
        if !tools.isEmpty {
            body["tools"] = tools.map { tool in
                var functionDict: [String: Any] = [
                    "name": tool.function.name,
                    "description": tool.function.description ?? ""
                ]
                
                if let parameters = tool.function.parameters {
                    if let jsonData = try? JSONEncoder().encode(parameters),
                       let jsonObject = try? JSONSerialization.jsonObject(with: jsonData) {
                        functionDict["parameters"] = jsonObject
                    } else {
                        functionDict["parameters"] = [String: Any]()
                    }
                } else {
                    functionDict["parameters"] = [String: Any]()
                }
                
                return [
                    "type": "function",
                    "function": functionDict
                ]
            }
            body["tool_choice"] = "auto"
        }
        
        var request = URLRequest(url: baseURL.appendingPathComponent("chat/completions"))
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        
        logger?.debug("Sending request to \(request.url?.absoluteString ?? "unknown")")
        
        return AsyncThrowingStream { continuation in
            let requestCopy = request
            Task {
                do {
                    let (bytes, response) = try await session.bytes(for: requestCopy)
                    
                    guard let httpResponse = response as? HTTPURLResponse else {
                        continuation.finish(throwing: LLMProviderError.invalidResponse)
                        return
                    }
                    
                    switch httpResponse.statusCode {
                    case 200...299:
                        break
                    case 401:
                        continuation.finish(throwing: LLMProviderError.authenticationError)
                        return
                    case 429:
                        continuation.finish(throwing: LLMProviderError.rateLimitExceeded)
                        return
                    default:
                        let errorMessage: String? = nil
                        continuation.finish(throwing: LLMProviderError.serverError(
                            httpResponse.statusCode,
                            errorMessage
                        ))
                        return
                    }
                    
                    for try await line in bytes.lines {
                        if line.isEmpty { continue }
                        
                        if line.hasPrefix("data: ") {
                            let data = String(line.dropFirst(6))
                            
                            if data == "[DONE]" {
                                continuation.finish()
                                return
                            }
                            
                            if let jsonData = data.data(using: .utf8) {
                                do {
                                    let chunk = try JSONDecoder().decode(
                                        ChatCompletionStreamOutput.self,
                                        from: jsonData
                                    )
                                    continuation.yield(chunk)
                                } catch {
                                    logger?.warning("Failed to decode chunk: \(error)")
                                }
                            }
                        }
                    }
                    
                    continuation.finish()
                } catch {
                    logger?.error("Stream error: \(error)")
                    continuation.finish(throwing: LLMProviderError.networkError(error))
                }
            }
        }
    }
}

extension OpenAICompatibleProvider {
    /// Create a provider for mlx_lm.server
    /// 
    /// mlx_lm.server provides an OpenAI-compatible API for running MLX models locally.
    /// Start the server with: `mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-4bit`
    ///
    /// - Parameters:
    ///   - model: The model identifier (must match the model loaded in mlx_lm.server)
    ///   - host: The server host (default: "localhost")
    ///   - port: The server port (default: 8080)
    ///   - logger: Optional logger for debugging
    public static func mlxServer(
        model: String,
        host: String = "localhost",
        port: Int = 8080,
        logger: Logger? = nil
    ) -> Self {
        .init(
            baseURL: URL(string: "http://\(host):\(port)/v1")!,
            apiKey: "mlx-server", // mlx_lm.server doesn't require an API key
            model: model,
            logger: logger
        )
    }
}