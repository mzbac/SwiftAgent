import Foundation
import MCP
import Logging

public typealias Value = MCP.Value

/// The main agent actor that orchestrates conversations between users, LLMs, and MCP tools.
///
/// `Agent` manages the conversation flow, handles tool execution, and maintains conversation state.
/// It supports multiple LLM providers and can connect to MCP servers for enhanced capabilities.
///
/// Example usage:
/// ```swift
/// let provider = OpenAICompatibleProvider.openAI(apiKey: "sk-...", model: "gpt-4")
/// let agent = Agent(
///     configuration: AgentConfiguration(
///         model: "gpt-4",
///         provider: "openai",
///         systemPrompt: "You are a helpful assistant."
///     ),
///     llmProvider: provider
/// )
///
/// try await agent.run("Hello!") { message in
///     print(message.content)
/// }
/// ```
public actor Agent {
    private let mcpClient: Client
    private let llmProvider: LLMProvider
    private let configuration: AgentConfiguration
    private let logger: Logger
    
    private var messages: [AgentMessage] = []
    private var isRunning = false
    private var currentTask: Task<Void, Swift.Error>?
    
    /// Creates a new agent with the specified configuration and LLM provider.
    /// - Parameters:
    ///   - configuration: The agent configuration including model, prompts, and limits
    ///   - llmProvider: The LLM provider to use for generating responses
    ///   - logger: Optional logger for debugging (defaults to SwiftAgent logger)
    public init(
        configuration: AgentConfiguration,
        llmProvider: LLMProvider,
        logger: Logger? = nil
    ) {
        self.configuration = configuration
        self.llmProvider = llmProvider
        self.logger = logger ?? Logger(label: "SwiftAgent")
        
        self.mcpClient = Client(
            name: "SwiftAgent",
            version: "1.0.0",
            configuration: .default
        )
        
        messages.append(AgentMessage(
            role: .system,
            content: configuration.systemPrompt
        ))
    }
    
    /// Connects to one or more MCP servers for tool access.
    /// - Parameter servers: Array of MCP server configurations
    /// - Throws: `AgentError.mcpConnectionFailed` if connection fails
    public func connect(servers: [MCPServerConfig]) async throws {
        logger.info("Connecting to \(servers.count) MCP servers")
        
        for server in servers {
            do {
                try await connectToServer(server)
                logger.info("Connected to server: \(server.name ?? server.type.rawValue)")
            } catch {
                logger.error("Failed to connect to server \(server.name ?? server.type.rawValue): \(error)")
                throw AgentError.mcpConnectionFailed(error)
            }
        }
    }
    
    /// Disconnects from all connected MCP servers.
    public func disconnect() async {
        logger.info("Disconnecting from MCP servers")
        await mcpClient.disconnect()
    }
    
    /// Returns the current conversation history.
    /// - Returns: Array of all messages in the conversation
    public func getMessages() -> [AgentMessage] {
        return messages
    }
    
    /// Clears the conversation history while preserving the system prompt.
    public func clearHistory() {
        messages = messages.filter { $0.role == .system }
    }
    
    /// Runs a conversation turn with the provided user input.
    /// 
    /// This method processes the user's message, generates responses using the LLM,
    /// executes any requested tools, and continues until the conversation completes
    /// or reaches the maximum number of turns.
    ///
    /// - Parameters:
    ///   - userInput: The user's message to process
    ///   - onMessage: Async callback invoked for each message in the conversation
    /// - Throws: `AgentError` if the conversation fails or is already running
    public func run(
        _ userInput: String,
        onMessage: @escaping (AgentMessage) async -> Void
    ) async throws {
        guard !isRunning else {
            throw AgentError.alreadyRunning
        }
        
        isRunning = true
        defer { isRunning = false }
        
        currentTask?.cancel()
        
        let task = Task {
            try await runConversation(userInput: userInput, onMessage: onMessage)
        }
        currentTask = task
        
        do {
            try await task.value
        } catch {
            if error is CancellationError {
                throw AgentError.cancelled
            }
            throw error
        }
    }
    
    /// Cancel the current agent run
    public func cancel() {
        currentTask?.cancel()
    }
    
    private func getMCPTools() async -> [Tool] {
        do {
            let (tools, _) = try await mcpClient.listTools()
            return tools
        } catch {
            logger.warning("Failed to list MCP tools: \(error)")
            return []
        }
    }
    
    
    private func connectToServer(_ config: MCPServerConfig) async throws {
        switch config.type {
        case .stdio:
            logger.warning("STDIO server connection requires launching the server process separately")
            throw AgentError.invalidServerConfig("STDIO servers need to be launched separately in Swift SDK")
            
        case .sse, .http:
            guard let url = config.url else {
                throw AgentError.invalidServerConfig("\(config.type.rawValue.uppercased()) server requires URL")
            }
            
            let transport = HTTPClientTransport(
                endpoint: url,
                streaming: config.streaming ?? true,
                logger: logger
            )
            
            _ = try await mcpClient.connect(transport: transport)
        }
    }
    
    private func runConversation(
        userInput: String,
        onMessage: @escaping (AgentMessage) async -> Void
    ) async throws {
        let userMessage = AgentMessage(role: .user, content: userInput)
        messages.append(userMessage)
        await onMessage(userMessage)
        
        var turnCount = 0
        var shouldContinue = true
        
        while shouldContinue && turnCount < configuration.maxTurns {
            try Task.checkCancellation()
            
            let mcpTools = await getMCPTools()
            let tools = mcpTools.asChatCompletionTools
            
            logger.debug("Starting turn \(turnCount + 1) with \(tools.count) available tools")
            
            let stream = try await llmProvider.complete(
                messages: messages,
                tools: tools,
                temperature: configuration.temperature,
                maxTokens: configuration.maxTokens,
                topP: configuration.topP,
                repetitionPenalty: configuration.repetitionPenalty,
                repetitionContextSize: configuration.repetitionContextSize
            )
            
            var assistantContent = ""
            var rawAssistantContent: String?
            var toolCalls: [ChatCompletionStreamOutputDeltaToolCall] = []
            
            do {
                for try await chunk in stream {
                    try Task.checkCancellation()
                    
                    if let delta = chunk.choices.first?.delta {
                        if let content = delta.content {
                            assistantContent += content
                        }
                        
                        if let rawContent = delta.rawContent {
                            rawAssistantContent = rawContent
                        }
                        
                        if let deltaToolCalls = delta.toolCalls {
                            for toolCall in deltaToolCalls {
                                if toolCall.index < toolCalls.count {
                                    var existing = toolCalls[toolCall.index]
                                    existing.function.arguments += toolCall.function.arguments
                                    toolCalls[toolCall.index] = existing
                                } else {
                                    toolCalls.append(toolCall)
                                }
                            }
                        }
                    }
                }
            } catch {
                logger.error("Error streaming LLM response: \(error)")
                throw AgentError.llmProviderError(error)
            }
            
            if !assistantContent.isEmpty {
                let assistantMessage = AgentMessage(
                    role: .assistant,
                    content: assistantContent,
                    rawContent: rawAssistantContent
                )
                messages.append(assistantMessage)
                await onMessage(assistantMessage)
            }
            
            if !toolCalls.isEmpty {
                logger.info("Processing \(toolCalls.count) tool calls")
                
                for toolCall in toolCalls {
                    try Task.checkCancellation()
                    
                    let toolResult = await executeToolCall(toolCall)
                    messages.append(toolResult)
                    await onMessage(toolResult)
                }
            } else {
                shouldContinue = false
            }
            
            turnCount += 1
        }
        
        if turnCount >= configuration.maxTurns {
            logger.warning("Maximum turns (\(configuration.maxTurns)) exceeded")
        }
    }
    
    private func executeToolCall(
        _ toolCall: ChatCompletionStreamOutputDeltaToolCall
    ) async -> AgentMessage {
        let toolName = toolCall.function.name
        logger.info("Executing tool: \(toolName)")
        logger.debug("Tool arguments: \(toolCall.function.arguments)")
        
        do {
            let arguments: [String: Value]?
            if !toolCall.function.arguments.isEmpty {
                let cleanedArgs = toolCall.function.arguments.trimmingCharacters(in: .whitespacesAndNewlines)
                let data = Data(cleanedArgs.utf8)
                
                // Try direct decoding first
                do {
                    arguments = try JSONDecoder().decode([String: Value].self, from: data)
                } catch let decodingError {
                    logger.debug("Failed to decode arguments directly: \(decodingError)")
                    logger.debug("Raw arguments data: \(String(data: data, encoding: .utf8) ?? "nil")")
                    
                    // Check if it's multiple JSON objects concatenated
                    if let rawString = String(data: data, encoding: .utf8),
                       rawString.contains("}{") {
                        // Try to extract just the first JSON object
                        if let firstBrace = rawString.firstIndex(of: "{"),
                           let closingBraceIndex = findMatchingClosingBrace(in: rawString, startingAt: firstBrace) {
                            let firstObject = String(rawString[firstBrace...closingBraceIndex])
                            logger.debug("Extracted first JSON object: \(firstObject)")
                            
                            if let firstObjectData = firstObject.data(using: .utf8),
                               let jsonObject = try? JSONSerialization.jsonObject(with: firstObjectData, options: []) as? [String: Any] {
                                arguments = jsonObject.mapValues { anyToValue($0) }
                            } else {
                                throw decodingError
                            }
                        } else {
                            throw decodingError
                        }
                    } else if let jsonObject = try? JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] {
                        arguments = jsonObject.mapValues { anyToValue($0) }
                    } else {
                        throw decodingError  // Re-throw the original decoding error
                    }
                }
            } else {
                arguments = nil
            }
            
            let result = try await mcpClient.callTool(
                name: toolName,
                arguments: arguments
            )
            
            let content = formatToolResult(result.content)
            
            return AgentMessage(
                role: .tool,
                content: content,
                toolCallId: toolCall.id,
                toolName: toolName
            )
        } catch {
            logger.error("Tool execution failed: \(error)")
            
            return AgentMessage(
                role: .tool,
                content: "Error executing tool '\(toolName)': \(error.localizedDescription)",
                toolCallId: toolCall.id,
                toolName: toolName
            )
        }
    }
    
    private func formatToolResult(_ content: [Tool.Content]) -> String {
        content.map { content in
            switch content {
            case .text(let text):
                return text
            case .image(let data, let mimeType, _):
                return "[Image: \(mimeType), \(estimateBase64Size(data)) bytes]"
            case .audio(let data, let mimeType):
                return "[Audio: \(mimeType), \(estimateBase64Size(data)) bytes]"
            case .resource(let uri, let mimeType, let text):
                if let text = text {
                    return text
                } else {
                    return "[Resource: \(uri), \(mimeType)]"
                }
            }
        }.joined(separator: "\n")
    }
    
    private func estimateBase64Size(_ base64String: String) -> Int {
        let data = base64String.split(separator: ",").last ?? Substring(base64String)
        let padding = data.suffix(2).filter { $0 == "=" }.count
        return (data.count * 3) / 4 - padding
    }
    
    private func anyToValue(_ any: Any) -> Value {
        switch any {
        case is NSNull:
            return .null
        case let bool as Bool:
            return .bool(bool)
        case let int as Int:
            return .int(int)
        case let double as Double:
            return .double(double)
        case let string as String:
            return .string(string)
        case let array as [Any]:
            return .array(array.map { anyToValue($0) })
        case let dict as [String: Any]:
            return .object(dict.mapValues { anyToValue($0) })
        default:
            // For any other type, convert to string
            return .string(String(describing: any))
        }
    }
    
    private func findMatchingClosingBrace(in string: String, startingAt startIndex: String.Index) -> String.Index? {
        var depth = 0
        var inString = false
        var escapeNext = false
        
        for index in string.indices[startIndex...] {
            let char = string[index]
            
            if escapeNext {
                escapeNext = false
                continue
            }
            
            if char == "\\" {
                escapeNext = true
                continue
            }
            
            if char == "\"" && !escapeNext {
                inString = !inString
                continue
            }
            
            if !inString {
                if char == "{" {
                    depth += 1
                } else if char == "}" {
                    depth -= 1
                    if depth == 0 {
                        return index
                    }
                }
            }
        }
        
        return nil
    }
}

extension Agent {
    /// Use the agent in an async context, automatically connecting and disconnecting
    public static func withAgent<T>(
        configuration: AgentConfiguration,
        llmProvider: LLMProvider,
        servers: [MCPServerConfig],
        logger: Logger? = nil,
        operation: (Agent) async throws -> T
    ) async throws -> T {
        let agent = Agent(
            configuration: configuration,
            llmProvider: llmProvider,
            logger: logger
        )
        
        try await agent.connect(servers: servers)
        
        do {
            let result = try await operation(agent)
            await agent.disconnect()
            return result
        } catch {
            await agent.disconnect()
            throw error
        }
    }
}