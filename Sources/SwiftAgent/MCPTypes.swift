import Foundation
import MCP

/// Represents a tool that can be called by the LLM
public struct ChatCompletionInputTool: Codable, Sendable {
    public let type: String
    public let function: Function
    
    public struct Function: Codable, Sendable {
        public let name: String
        public let description: String?
        public let parameters: Value?
        
        public init(name: String, description: String? = nil, parameters: Value? = nil) {
            self.name = name
            self.description = description
            self.parameters = parameters
        }
    }
    
    public init(type: String = "function", function: Function) {
        self.type = type
        self.function = function
    }
    
    public static func from(tool: Tool) -> ChatCompletionInputTool {
        return ChatCompletionInputTool(
            function: Function(
                name: tool.name,
                description: tool.description,
                parameters: tool.inputSchema
            )
        )
    }
}

/// Represents the streaming output from an LLM chat completion
public struct ChatCompletionStreamOutput: Codable, Sendable {
    public let id: String
    public let choices: [ChatCompletionStreamOutputChoice]
    public let created: Date
    public let model: String
    public let systemFingerprint: String?
    
    public init(
        id: String,
        choices: [ChatCompletionStreamOutputChoice],
        created: Date,
        model: String,
        systemFingerprint: String? = nil
    ) {
        self.id = id
        self.choices = choices
        self.created = created
        self.model = model
        self.systemFingerprint = systemFingerprint
    }
}

/// Represents a choice in the streaming output
public struct ChatCompletionStreamOutputChoice: Codable, Sendable {
    public let index: Int
    public let delta: ChatCompletionStreamOutputDelta
    public let finishReason: String?
    
    private enum CodingKeys: String, CodingKey {
        case index, delta
        case finishReason = "finish_reason"
    }
    
    public init(index: Int, delta: ChatCompletionStreamOutputDelta, finishReason: String? = nil) {
        self.index = index
        self.delta = delta
        self.finishReason = finishReason
    }
}

/// Represents the delta (incremental update) in streaming output
public struct ChatCompletionStreamOutputDelta: Codable, Sendable {
    public let role: String?
    public let content: String?
    public let tool_calls: [ChatCompletionStreamOutputDeltaToolCall]?
    
    public init(role: String? = nil, content: String? = nil, tool_calls: [ChatCompletionStreamOutputDeltaToolCall]? = nil) {
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
    }
}

/// Represents a tool call in the streaming output delta
public struct ChatCompletionStreamOutputDeltaToolCall: Codable, Sendable {
    public let index: Int
    public let id: String
    public var function: ChatCompletionStreamOutputDeltaToolCallFunction
    
    public init(index: Int, id: String, function: ChatCompletionStreamOutputDeltaToolCallFunction) {
        self.index = index
        self.id = id
        self.function = function
    }
}

/// Represents the function details in a tool call
public struct ChatCompletionStreamOutputDeltaToolCallFunction: Codable, Sendable {
    public let name: String
    public var arguments: String
    
    public init(name: String, arguments: String) {
        self.name = name
        self.arguments = arguments
    }
}

extension Array where Element == Tool {
    public var asChatCompletionTools: [ChatCompletionInputTool] {
        self.map { ChatCompletionInputTool.from(tool: $0) }
    }
}