import Foundation

/// Configuration for MCP servers
public struct MCPServerConfig: Sendable, Codable {
    public enum ServerType: String, Sendable, Codable {
        case stdio
        case sse
        case http
    }
    
    public let type: ServerType
    public let name: String?
    public let command: String?
    public let args: [String]?
    public let env: [String: String]?
    public let cwd: String?
    public let url: URL?
    public let headers: [String: String]?
    public let streaming: Bool?
    
    private init(
        type: ServerType,
        name: String? = nil,
        command: String? = nil,
        args: [String]? = nil,
        env: [String: String]? = nil,
        cwd: String? = nil,
        url: URL? = nil,
        headers: [String: String]? = nil,
        streaming: Bool? = nil
    ) {
        self.type = type
        self.name = name
        self.command = command
        self.args = args
        self.env = env
        self.cwd = cwd
        self.url = url
        self.headers = headers
        self.streaming = streaming
    }
    
    /// Create a stdio server configuration
    public static func stdio(
        name: String? = nil,
        command: String,
        args: [String] = [],
        env: [String: String]? = nil,
        cwd: String? = nil
    ) -> Self {
        .init(
            type: .stdio,
            name: name,
            command: command,
            args: args,
            env: env,
            cwd: cwd
        )
    }
    
    /// Create an SSE server configuration
    public static func sse(
        name: String? = nil,
        url: URL,
        headers: [String: String]? = nil
    ) -> Self {
        .init(
            type: .sse,
            name: name,
            url: url,
            headers: headers
        )
    }
    
    /// Create an HTTP server configuration
    public static func http(
        name: String? = nil,
        url: URL,
        headers: [String: String]? = nil,
        streaming: Bool = true
    ) -> Self {
        .init(
            type: .http,
            name: name,
            url: url,
            headers: headers,
            streaming: streaming
        )
    }
}

// MARK: - Common Server Configurations

extension MCPServerConfig {
    /// File system access server
    public static func fileSystem(path: String = NSHomeDirectory()) -> Self {
        .stdio(
            name: "filesystem",
            command: "npx",
            args: ["-y", "@modelcontextprotocol/server-filesystem", path]
        )
    }
    
    /// Git server for repository operations
    public static func git(repository: String? = nil) -> Self {
        var args = ["-y", "@modelcontextprotocol/server-git"]
        if let repo = repository {
            args.append(repo)
        }
        
        return .stdio(
            name: "git",
            command: "npx",
            args: args
        )
    }
    
    /// Playwright browser automation server
    public static func playwright() -> Self {
        .stdio(
            name: "playwright",
            command: "npx",
            args: ["-y", "@playwright/mcp"]
        )
    }
}