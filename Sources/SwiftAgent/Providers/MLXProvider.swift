import Foundation
import MLX
import MLXNN
@preconcurrency import MLXLMCommon
import MLXLLM
import Hub
import Tokenizers
import MCP
import Logging

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

public actor MLXProvider: LLMProvider {
    private let modelContainer: ModelContainer
    private let modelName: String
    private let logger: Logger
    
    private struct PromptCache {
        let modelKey: String
        var cachedTokens: [Int]
        var kvCache: [KVCache]
        var lastMessageCount: Int
        var lastMessages: [AgentMessage]
        let createdAt: Date
        var lastAccessedAt: Date
        
        var isValid: Bool {
            Date().timeIntervalSince(lastAccessedAt) < 1800
        }
        
        mutating func updateAccess() {
            lastAccessedAt = Date()
        }
    }
    
    private var promptCache: PromptCache?
    
    private struct CacheStats {
        var hits: Int = 0
        var misses: Int = 0
        var partialHits: Int = 0
        var resets: Int = 0
        var trims: Int = 0
        
        var hitRate: Double {
            let total = hits + misses
            return total > 0 ? Double(hits) / Double(total) : 0
        }
    }
    
    private var cacheStats = CacheStats()
    
    private func commonPrefixLength(_ tokens1: [Int], _ tokens2: [Int]) -> Int {
        let minLength = min(tokens1.count, tokens2.count)
        var mismatchIndex = -1
        for i in 0..<minLength {
            if tokens1[i] != tokens2[i] {
                mismatchIndex = i
                break
            }
        }
        
        if mismatchIndex == -1 {
            mismatchIndex = minLength
        }
        
        return mismatchIndex
    }
    
    private func resetPromptCache(modelKey: String, tokens: [Int], kvCache: [KVCache], messages: [AgentMessage]) {
        promptCache = PromptCache(
            modelKey: modelKey,
            cachedTokens: tokens,
            kvCache: kvCache,
            lastMessageCount: messages.count,
            lastMessages: messages,
            createdAt: Date(),
            lastAccessedAt: Date()
        )
        cacheStats.resets += 1
    }
    
    private func getPromptCache(
        modelKey: String,
        messageCount: Int,
        newTokens: [Int]
    ) async -> (tokensToProcess: [Int], cacheToUse: [KVCache]?, startIndex: Int, cachedMessages: Int) {
        guard var cache = promptCache,
              cache.modelKey == modelKey,
              cache.isValid else {
            cacheStats.misses += 1
            logger.debug("Prompt cache miss: cache invalid or different model")
            return (newTokens, nil, 0, 0)
        }
        
        cache.updateAccess()
        promptCache = cache
        let commonLen = commonPrefixLength(cache.cachedTokens, newTokens)
        let effectiveCommonLen = min(commonLen, newTokens.count - 1)
        
        if effectiveCommonLen == 0 {
            cacheStats.misses += 1
            logger.debug("Prompt cache miss: no common prefix found")
            return (newTokens, nil, 0, 0)
        }
        
        let cacheLen = cache.cachedTokens.count
        
        if effectiveCommonLen == cacheLen {
            cacheStats.hits += 1
            logger.debug("Prompt cache hit: reusing \(effectiveCommonLen) tokens (\(String(format: "%.1f%%", Double(effectiveCommonLen) / Double(newTokens.count) * 100)) of prompt)")
            let tokensToProcess = Array(newTokens[effectiveCommonLen...])
            return (tokensToProcess, cache.kvCache, effectiveCommonLen, cache.lastMessageCount)
        }
        
        if effectiveCommonLen < cacheLen {
            cacheStats.partialHits += 1
            
            let kvCache = cache.kvCache
            if kvCache.allSatisfy({ $0.isTrimmable }) {
                let trimAmount = cacheLen - effectiveCommonLen
                
                for cache in kvCache {
                    _ = cache.trim(trimAmount)
                }
                
                cache.cachedTokens = Array(newTokens.prefix(effectiveCommonLen))
                cache.updateAccess()
                promptCache = cache
                
                let tokensToProcess = Array(newTokens[effectiveCommonLen...])
                cacheStats.trims += 1
                cacheStats.hits += 1
                logger.debug("Prompt cache partial hit: trimmed cache from \(cacheLen) to \(effectiveCommonLen) tokens, reusing \(String(format: "%.1f%%", Double(effectiveCommonLen) / Double(newTokens.count) * 100)) of prompt")
                return (tokensToProcess, kvCache, effectiveCommonLen, cache.lastMessageCount)
            } else {
                cacheStats.misses += 1
                return (newTokens, nil, 0, 0)
            }
        }
        
        cacheStats.misses += 1
        return (newTokens, nil, 0, 0)
    }
    
    private func updatePromptCache(
        modelKey: String,
        allTokens: [Int],
        newKvCache: [KVCache],
        messages: [AgentMessage]
    ) {
        resetPromptCache(
            modelKey: modelKey,
            tokens: allTokens,
            kvCache: newKvCache,
            messages: messages
        )
    }
    
    private func logCacheStats() {
        logger.info("""
            Cache Statistics:
            - Hits: \(cacheStats.hits)
            - Misses: \(cacheStats.misses)
            - Partial Hits: \(cacheStats.partialHits)
            - Hit Rate: \(String(format: "%.2f%%", cacheStats.hitRate * 100))
            - Resets: \(cacheStats.resets)
            - Trims: \(cacheStats.trims)
            """)
    }
    
    public init(modelId: String, downloadPath: URL? = nil, quantization: BaseConfiguration.Quantization? = nil) async throws {
        self.modelName = modelId
        self.logger = Logger(label: "SwiftAgent.MLXProvider")
        
        let hub = downloadPath.map { HubApi(downloadBase: $0) } ?? HubApi()
        
        let modelFactory = LLMModelFactory.shared
        let configuration = modelFactory.configuration(id: modelId)
        
        let context = try await modelFactory.load(
            hub: hub,
            configuration: configuration,
            progressHandler: { progress in
                Logger(label: "SwiftAgent.MLXProvider").info("Download progress: \(progress.fractionCompleted * 100)%")
            }
        )
        
        let modelDirectory = configuration.modelDirectory(hub: hub)
        let configURL = modelDirectory.appending(component: "config.json")
        if FileManager.default.fileExists(atPath: configURL.path) {
            let configData = try Data(contentsOf: configURL)
            let baseConfig = try JSONDecoder().decode(BaseConfiguration.self, from: configData)

            guard baseConfig.modelType == "qwen3" || baseConfig.modelType == "qwen3_moe" else {
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
                    let modelKey = "\(modelName)-\(temperature)-\(topP ?? 0.95)"
                    
                    self.invalidateCacheIfNeeded(forMessageCount: messages.count)
                    
                    let processedMessages = messages
                    
                    let tokenizationData = try await self.modelContainer.perform { context in
                        let tokenizer = context.tokenizer
                        
                        let toolsJSON: String? = if tools.isEmpty {
                            nil
                        } else {
                            try tools.sorted(by: { $0.function.name < $1.function.name }).map { tool in
                                let parameters = MLXProvider.convertValueToAny(tool.function.parameters)
                                
                                let functionDict: [String: Any] = [
                                    "description": tool.function.description ?? "",
                                    "name": tool.function.name,
                                    "parameters": parameters
                                ]
                                
                                let toolSpec: [String: Any] = [
                                    "function": functionDict,
                                    "type": "function"
                                ]
                                
                                return try MLXProvider.canonicalJSONString(from: toolSpec)
                            }.joined(separator: "\n")
                        }
                        
                        var prompt = ""
                        
                        if let systemMessage = processedMessages.first(where: { $0.role == .system }) {
                            prompt += "<|im_start|>system\n"
                            prompt += systemMessage.tokenizableContent
                            
                            if let toolsJSON = toolsJSON {
                                prompt += "\n\n# Tools\n\n"
                                prompt += "You may call one or more functions to assist with the user query.\n\n"
                                prompt += "You are provided with function signatures within <tools></tools> XML tags:\n"
                                prompt += "<tools>\n"
                                prompt += toolsJSON
                                prompt += "\n</tools>\n\n"
                                prompt += "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
                                prompt += "<tool_call>\n"
                                prompt += "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
                                prompt += "</tool_call>"
                            }
                            
                            prompt += "<|im_end|>\n"
                        }
                        
                        for message in processedMessages where message.role != .system {
                            switch message.role {
                            case .user:
                                prompt += "<|im_start|>user\n"
                                prompt += message.tokenizableContent
                                prompt += "<|im_end|>\n"
                            case .assistant:
                                prompt += "<|im_start|>assistant\n"
                                prompt += message.tokenizableContent
                                prompt += "<|im_end|>\n"
                            case .tool:
                                prompt += "<|im_start|>user\n"
                                prompt += "<tool_response>\n"
                                prompt += message.tokenizableContent
                                prompt += "\n</tool_response>"
                                prompt += "<|im_end|>\n"
                            default:
                                break
                            }
                        }
                        
                        prompt += "<|im_start|>assistant\n"
                        
                        let fullTokenIds = tokenizer.encode(text: prompt)
                        
                        return (
                            context: context,
                            fullTokenIds: fullTokenIds
                        )
                    }
                    
                    let cacheInfo = await self.getPromptCache(
                        modelKey: modelKey,
                        messageCount: messages.count,
                        newTokens: tokenizationData.fullTokenIds
                    )
                    
                    let tokensToProcess = cacheInfo.tokensToProcess
                    let kvCache = cacheInfo.cacheToUse
                    let startIndex = cacheInfo.startIndex
                    let cachedMessages = cacheInfo.cachedMessages
                    
                    self.logger.info("Cache info - Total tokens: \(tokenizationData.fullTokenIds.count), Processing: \(tokensToProcess.count), Cached: \(startIndex), Cached messages: \(cachedMessages)")
                                        
                    let generationData = try await self.modelContainer.perform { context in
                        let lmInput: LMInput
                        if kvCache != nil && !tokensToProcess.isEmpty {
                            let promptTokens = MLXArray(tokensToProcess)
                            lmInput = LMInput(tokens: promptTokens)
                        } else {
                            lmInput = LMInput(tokens: MLXArray(tokenizationData.fullTokenIds))
                        }
                        
                        let currentCache = kvCache ?? context.model.newCache(parameters: nil)
                        
                        return (
                            context: context,
                            lmInput: lmInput,
                            currentCache: currentCache
                        )
                    }
                    
                    let stream = try generate(
                        input: generationData.lmInput,
                        cache: generationData.currentCache,
                        parameters: parameters,
                        context: generationData.context
                    )
                    
                    var rawResponse = ""
                    var promptTokensPerSec: Double = 0
                    var generationTokensPerSec: Double = 0
                    
                    for try await generation in stream {
                        switch generation {
                        case .chunk(let text):
                            rawResponse += text
                        case .info(let info):
                            promptTokensPerSec = info.promptTokensPerSecond
                            generationTokensPerSec = info.tokensPerSecond
                            break
                        }
                    }
                    
                    
                    self.updatePromptCache(
                        modelKey: modelKey,
                        allTokens: tokenizationData.fullTokenIds,
                        newKvCache: generationData.currentCache,
                        messages: processedMessages
                    )
                    
                    let totalRequests = self.cacheStats.hits + self.cacheStats.misses
                    if totalRequests > 0 && totalRequests % 10 == 0 {
                        self.logCacheStats()
                    }
                    
                    var filteredResponse = rawResponse
                    while let thinkStart = filteredResponse.range(of: "<think>"),
                          let thinkEnd = filteredResponse.range(of: "</think>", range: thinkStart.upperBound..<filteredResponse.endIndex) {
                        filteredResponse.removeSubrange(thinkStart.lowerBound...thinkEnd.upperBound)
                    }
                    
                    var toolCalls: [ChatCompletionStreamOutputDeltaToolCall] = []
                    var contentBeforeTools = filteredResponse.trimmingCharacters(in: .whitespacesAndNewlines)
                    
                    if rawResponse.contains("tool") || rawResponse.contains("function") || rawResponse.contains("<tool_call>") {
                        logger.trace("Model response contains tool-related keywords")
                    }
                    
                    if rawResponse.contains("<tool_call>") {
                        logger.trace("Full response contains tool call. Response length: \(rawResponse.count)")
                        
                        if let range = rawResponse.range(of: "<tool_call>") {
                            contentBeforeTools = String(rawResponse[..<range.lowerBound])
                            
                            while let thinkStart = contentBeforeTools.range(of: "<think>"),
                                  let thinkEnd = contentBeforeTools.range(of: "</think>", range: thinkStart.upperBound..<contentBeforeTools.endIndex) {
                                contentBeforeTools.removeSubrange(thinkStart.lowerBound...thinkEnd.upperBound)
                            }
                            
                            contentBeforeTools = contentBeforeTools.trimmingCharacters(in: .whitespacesAndNewlines)
                            logger.trace("Content before tools (filtered): \(contentBeforeTools)")
                        }
                        
                        var remainingText = rawResponse
                        while let startRange = remainingText.range(of: "<tool_call>") {
                            logger.trace("Found <tool_call> at index: \(remainingText.distance(from: remainingText.startIndex, to: startRange.lowerBound))")
                            
                            if let endRange = remainingText.range(of: "</tool_call>", range: startRange.upperBound..<remainingText.endIndex) {
                                logger.trace("Found </tool_call> at index: \(remainingText.distance(from: remainingText.startIndex, to: endRange.lowerBound))")
                                
                                let endIndex = remainingText.index(endRange.upperBound, offsetBy: 1, limitedBy: remainingText.endIndex) ?? remainingText.endIndex
                                let toolCallText = String(remainingText[startRange.lowerBound..<endIndex])
                                logger.trace("Found complete tool call")
                                
                                if let toolCall = self.parseToolCall(from: toolCallText) {
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
                                    content: toolCalls.isEmpty ? filteredResponse : contentBeforeTools,
                                    rawContent: rawResponse,
                                    toolCalls: toolCalls.isEmpty ? nil : toolCalls
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
                                    toolCalls: nil
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
                    continuation.finish(throwing: self.mapMLXError(error))
                }
            }
        }
    }
    
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
    
    public func clearCache() {
        promptCache = nil
    }
    
    public func getCacheStatistics() -> (hits: Int, misses: Int, hitRate: Double) {
        return (
            hits: cacheStats.hits,
            misses: cacheStats.misses,
            hitRate: cacheStats.hitRate
        )
    }
    
    public func hasCacheAvailable() -> Bool {
        return promptCache?.isValid ?? false
    }
    
    public func invalidateCacheIfNeeded(forMessageCount newCount: Int) {
        guard let cache = promptCache else { return }
        
        if newCount < cache.lastMessageCount {
            clearCache()
            cacheStats.resets += 1
        }
    }
    
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
    
    
    /// Converts an object to a canonical JSON string with sorted keys
    private static func canonicalJSONString(from object: Any) throws -> String {
        let data = try JSONSerialization.data(
            withJSONObject: object,
            options: [.sortedKeys, .withoutEscapingSlashes]
        )
        return String(data: data, encoding: .utf8)!
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