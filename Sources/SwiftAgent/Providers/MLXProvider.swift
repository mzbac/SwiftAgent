import Foundation
import MLX
import MLXNN
@preconcurrency import MLXLMCommon
import MLXLLM
import Hub
import Tokenizers
import MCP
import Logging

/// Error types specific to MLX provider
public enum MLXProviderError: LocalizedError {
    case modelLoadingFailed(String)
    case inferenceError(String)
    case unsupportedFeature(String)
    
    public var errorDescription: String? {
        switch self {
        case .modelLoadingFailed(let message):
            return "Failed to load model: \(message)"
        case .inferenceError(let message):
            return "Inference error: \(message)"
        case .unsupportedFeature(let message):
            return "Unsupported feature: \(message)"
        }
    }
}

/// A provider that runs models locally using MLX Swift
public actor MLXProvider: LLMProvider {
    private let modelContainer: ModelContainer
    private let modelName: String
    private let logger: Logger
    
    /// Initialize with a HuggingFace model ID
    public init(modelId: String, downloadPath: URL? = nil, quantization: BaseConfiguration.Quantization? = nil) async throws {
        self.modelName = modelId
        self.logger = Logger(label: "SwiftAgent.MLXProvider")
        
        // Create hub API with optional custom download path
        let hub = downloadPath.map { HubApi(downloadBase: $0) } ?? HubApi()
        
        // Create model configuration
        let modelFactory = LLMModelFactory.shared
        let configuration = modelFactory.configuration(id: modelId)
        
        // Load model and tokenizer
        let context = try await modelFactory.load(
            hub: hub,
            configuration: configuration,
            progressHandler: { progress in
                Logger(label: "SwiftAgent.MLXProvider").info("Download progress: \(progress.fractionCompleted * 100)%")
            }
        )
        
        // Check if the model is Qwen3 (for now only support Qwen3)
        let modelDirectory = configuration.modelDirectory(hub: hub)
        let configURL = modelDirectory.appending(component: "config.json")
        if FileManager.default.fileExists(atPath: configURL.path) {
            let configData = try Data(contentsOf: configURL)
            let baseConfig = try JSONDecoder().decode(BaseConfiguration.self, from: configData)
            
            // Check if model type is qwen3
            guard baseConfig.modelType == "qwen3" else {
                throw MLXProviderError.unsupportedFeature(
                    "Only Qwen3 models are supported for now. Found model type: \(baseConfig.modelType)"
                )
            }
        }
        
        self.modelContainer = ModelContainer(context: context)
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
        
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    let parameters = GenerateParameters(
                        maxTokens: maxTokens,
                        temperature: Float(temperature),
                        topP: Float(topP ?? 0.95),
                        repetitionPenalty: Float(repetitionPenalty ?? 1.15),
                        repetitionContextSize: repetitionContextSize ?? 20,
                        kvBits: 8
                    )
                    
                    let completionId = UUID().uuidString
                    
                    let fullResponse = try await self.modelContainer.perform { context in
                        let processedMessages = messages
                        
                        let messageDicts: [[String: Any]] = processedMessages.map { message in
                            let dict: [String: Any] = [
                                "role": message.role.rawValue,
                                "content": message.content
                            ]
                            return dict
                        }
                        
                        let tokenizer = context.tokenizer
                        
                        let toolSpecs: [[String: Any]]? = tools.isEmpty ? nil : tools.map { tool in
                            [
                                "type": "function",
                                "function": [
                                    "name": tool.function.name,
                                    "description": tool.function.description ?? "",
                                    "parameters": MLXProvider.convertValueToAny(tool.function.parameters)
                                ]
                            ]
                        }
                        
                        let tokenIds: [Int]
                        if let toolSpecs = toolSpecs {
                            tokenIds = try tokenizer.applyChatTemplate(
                                messages: messageDicts,
                                tools: toolSpecs
                            )
                        } else {
                            tokenIds = try tokenizer.applyChatTemplate(
                                messages: messageDicts
                            )
                        }
                        
                        logger.trace("Applied chat template, got \(tokenIds.count) tokens")
                        
                        let promptText = tokenizer.decode(tokens: tokenIds)
                        logger.trace("Decoded prompt text length: \(promptText.count)")
                        
                        let userInput = UserInput(prompt: .text(promptText))
                        
                        let lmInput = try await context.processor.prepare(input: userInput)
                        
                        let stream = try generate(
                            input: lmInput,
                            parameters: parameters,
                            context: context
                        )
                        
                        var response = ""
                        for try await generation in stream {
                            switch generation {
                            case .chunk(let text):
                                response += text
                            case .info(_):
                                break
                            }
                        }
                        
                        logger.trace("Generation complete. Response length: \(response.count)")
                        
                        var filteredResponse = response
                        while let thinkStart = filteredResponse.range(of: "<think>"),
                              let thinkEnd = filteredResponse.range(of: "</think>", range: thinkStart.upperBound..<filteredResponse.endIndex) {
                            filteredResponse.removeSubrange(thinkStart.lowerBound...thinkEnd.upperBound)
                        }
                        
                        return filteredResponse
                    }
                    
                    var toolCalls: [ChatCompletionStreamOutputDeltaToolCall] = []
                    var contentBeforeTools = fullResponse.trimmingCharacters(in: .whitespacesAndNewlines)
                    
                    if fullResponse.contains("tool") || fullResponse.contains("function") || fullResponse.contains("<tool_call>") {
                        logger.trace("Model response contains tool-related keywords")
                    }
                    
                    if fullResponse.contains("<tool_call>") {
                        logger.trace("Full response contains tool call. Response length: \(fullResponse.count)")
                        
                        if let range = fullResponse.range(of: "<tool_call>") {
                            contentBeforeTools = String(fullResponse[..<range.lowerBound])
                            
                            while let thinkStart = contentBeforeTools.range(of: "<think>"),
                                  let thinkEnd = contentBeforeTools.range(of: "</think>", range: thinkStart.upperBound..<contentBeforeTools.endIndex) {
                                contentBeforeTools.removeSubrange(thinkStart.lowerBound...thinkEnd.upperBound)
                            }
                            
                            contentBeforeTools = contentBeforeTools.trimmingCharacters(in: .whitespacesAndNewlines)
                            logger.trace("Content before tools (filtered): \(contentBeforeTools)")
                        }
                        
                        var remainingText = fullResponse
                        while let startRange = remainingText.range(of: "<tool_call>") {
                            logger.trace("Found <tool_call> at index: \(remainingText.distance(from: remainingText.startIndex, to: startRange.lowerBound))")
                            
                            if let endRange = remainingText.range(of: "</tool_call>", range: startRange.upperBound..<remainingText.endIndex) {
                                logger.trace("Found </tool_call> at index: \(remainingText.distance(from: remainingText.startIndex, to: endRange.lowerBound))")
                                
                                let endIndex = remainingText.index(endRange.upperBound, offsetBy: 1, limitedBy: remainingText.endIndex) ?? remainingText.endIndex
                                let toolCallText = String(remainingText[startRange.lowerBound..<endIndex])
                                logger.trace("Found complete tool call")
                                
                                if let toolCall = parseToolCall(from: toolCallText) {
                                    toolCalls.append(toolCall)
                                }
                                
                                let nextIndex = remainingText.index(endRange.upperBound, offsetBy: 1, limitedBy: remainingText.endIndex) ?? remainingText.endIndex
                                if nextIndex < remainingText.endIndex {
                                    remainingText = String(remainingText[nextIndex...])
                                    logger.trace("Remaining text after tool call")
                                } else {
                                    logger.trace("No more text after tool call")
                                    break
                                }
                            } else {
                                logger.trace("No closing </tool_call> tag found")
                                break
                            }
                        }
                    }
                    
                    let chunk = ChatCompletionStreamOutput(
                        id: completionId,
                        choices: [
                            ChatCompletionStreamOutputChoice(
                                index: 0,
                                delta: ChatCompletionStreamOutputDelta(
                                    role: "assistant",
                                    content: toolCalls.isEmpty ? fullResponse : contentBeforeTools,
                                    tool_calls: toolCalls.isEmpty ? nil : toolCalls
                                ),
                                finishReason: nil
                            )
                        ],
                        created: Date(),
                        model: modelName,
                        systemFingerprint: nil
                    )
                    continuation.yield(chunk)
                    
                    let finalChunk = ChatCompletionStreamOutput(
                        id: completionId,
                        choices: [
                            ChatCompletionStreamOutputChoice(
                                index: 0,
                                delta: ChatCompletionStreamOutputDelta(
                                    role: nil,
                                    content: nil,
                                    tool_calls: nil
                                ),
                                finishReason: toolCalls.isEmpty ? "stop" : "tool_calls"
                            )
                        ],
                        created: Date(),
                        model: modelName,
                        systemFingerprint: nil
                    )
                    continuation.yield(finalChunk)
                    
                    continuation.finish()
                    
                } catch {
                    continuation.finish(throwing: mapMLXError(error))
                }
            }
        }
    }
    
    /// Map MLX errors to MLXProviderError
    private func mapMLXError(_ error: Swift.Error) -> Swift.Error {
        let nsError = error as NSError
        switch nsError.domain {
        case "MLXError":
            return MLXProviderError.inferenceError(error.localizedDescription)
        case NSURLErrorDomain:
            return MLXProviderError.modelLoadingFailed("Network error: \(error.localizedDescription)")
        default:
            return MLXProviderError.inferenceError("Unexpected error: \(error.localizedDescription)")
        }
    }
    
    /// Convert MCP Value type to standard Swift types
    private static func convertValueToAny(_ value: Value?) -> Any {
        guard let value = value else { return [:] }
        
        switch value {
        case .null:
            return NSNull()
        case .bool(let b):
            return b
        case .int(let i):
            return i
        case .double(let d):
            return d
        case .string(let s):
            return s
        case .data(_, let data):
            return data.base64EncodedString()
        case .array(let arr):
            return arr.map { convertValueToAny($0) }
        case .object(let obj):
            var dict: [String: Any] = [:]
            for (key, val) in obj {
                dict[key] = convertValueToAny(val)
            }
            return dict
        }
    }
    
    private func parseToolCall(from text: String) -> ChatCompletionStreamOutputDeltaToolCall? {
        guard let startRange = text.range(of: "<tool_call>"),
              let endRange = text.range(of: "</tool_call>"),
              startRange.upperBound < endRange.lowerBound else {
            logger.trace("Failed to find tool call tags")
            return nil
        }
        
        let jsonContent = String(text[startRange.upperBound..<endRange.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        
        logger.trace("Extracted JSON content: \(jsonContent)")
        
        guard let jsonData = jsonContent.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any],
              let name = json["name"] as? String else {
            logger.trace("Failed to parse JSON from content")
            return nil
        }
        
        let arguments: String
        if let args = json["arguments"] {
            if let argsString = args as? String {
                arguments = argsString
            } else if let argsData = try? JSONSerialization.data(withJSONObject: args, options: []),
                      let argsString = String(data: argsData, encoding: .utf8) {
                arguments = argsString
            } else {
                arguments = "{}"
            }
        } else {
            arguments = "{}"
        }
        
        logger.trace("Parsed tool call - name: \(name), arguments: \(arguments)")
        
        return ChatCompletionStreamOutputDeltaToolCall(
            index: 0,
            id: UUID().uuidString,
            function: ChatCompletionStreamOutputDeltaToolCallFunction(
                name: name,
                arguments: arguments
            )
        )
    }
}