import Foundation
import MLXLMCommon

/// Utilities for converting between SwiftAgent and MLX message formats
extension AgentMessage.Role {
    /// Convert SwiftAgent role to MLX Chat.Message.Role
    var toMLXRole: Chat.Message.Role {
        switch self {
        case .system:
            return .system
        case .user:
            return .user
        case .assistant:
            return .assistant
        case .tool:
            return .user
        }
    }
}

extension Chat.Message.Role {
    /// Convert MLX role to SwiftAgent role
    var toAgentRole: AgentMessage.Role {
        switch self {
        case .system:
            return .system
        case .user:
            return .user
        case .assistant:
            return .assistant
        }
    }
}

extension AgentMessage {
    /// Convert AgentMessage to MLX Chat.Message
    func toMLXMessage() -> Chat.Message {
        var content = self.content
        
        if role == .tool {
            content = "<tool_response>\n\(content)\n</tool_response>"
        }
        
        return Chat.Message(
            role: role.toMLXRole,
            content: content,
            images: [],
            videos: []
        )
    }
}

extension Chat.Message {
    /// Convert MLX Chat.Message to AgentMessage
    func toAgentMessage() -> AgentMessage {
        if role == .user && content.contains("<tool_response>") {
            if let startRange = content.range(of: "<tool_response>\n"),
               let endRange = content.range(of: "\n</tool_response>") {
                let toolContent = String(content[startRange.upperBound..<endRange.lowerBound])
                return AgentMessage(role: .tool, content: toolContent)
            }
        }
        
        return AgentMessage(
            role: role.toAgentRole,
            content: content
        )
    }
}

/// Convert array of AgentMessages to MLX format
extension Array where Element == AgentMessage {
    func toMLXMessages() -> [Chat.Message] {
        self.map { $0.toMLXMessage() }
    }
}

/// Convert array of MLX Chat.Messages to AgentMessage format
extension Array where Element == Chat.Message {
    func toAgentMessages() -> [AgentMessage] {
        self.map { $0.toAgentMessage() }
    }
}