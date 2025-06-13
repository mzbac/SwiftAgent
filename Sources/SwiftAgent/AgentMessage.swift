import Foundation

/// Represents a message in the conversation
public struct AgentMessage: Sendable, Identifiable, Codable {
    public enum Role: String, Sendable, Codable {
        case user
        case assistant
        case system
        case tool
    }
    
    public let id: UUID
    public let role: Role
    public let content: String
    public let rawContent: String?
    public let toolCallId: String?
    public let toolName: String?
    public let timestamp: Date
    
    public init(
        id: UUID = UUID(),
        role: Role,
        content: String,
        rawContent: String? = nil,
        toolCallId: String? = nil,
        toolName: String? = nil,
        timestamp: Date = Date()
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.rawContent = rawContent
        self.toolCallId = toolCallId
        self.toolName = toolName
        self.timestamp = timestamp
    }
}

// MARK: - Extensions

extension AgentMessage {
    public var tokenizableContent: String {
        rawContent ?? content
    }
    /// Convert to OpenAI API format
    public func toOpenAIFormat() -> [String: Any] {
        var dict: [String: Any] = [
            "role": role.rawValue,
            "content": content
        ]
        
        if let toolCallId = toolCallId {
            dict["tool_call_id"] = toolCallId
        }
        
        if let toolName = toolName {
            dict["name"] = toolName
        }
        
        return dict
    }
}