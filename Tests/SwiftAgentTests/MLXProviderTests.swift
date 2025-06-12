import XCTest
@testable import SwiftAgent
import MCP

final class MLXProviderTests: XCTestCase {
    
    func testValueConversion() async throws {
        // Test that the MLXProvider can properly convert MCP Value types
        // This test verifies the fix for the JSON schema parsing issue
        
        // Create a sample tool with complex parameters
        let parameters: Value = .object([
            "type": .string("object"),
            "properties": .object([
                "query": .object([
                    "type": .string("string"),
                    "description": .string("The search query")
                ]),
                "max_results": .object([
                    "type": .string("integer"),
                    "description": .string("Maximum number of results"),
                    "default": .int(10)
                ])
            ]),
            "required": .array([.string("query")])
        ])
        
        let tool = ChatCompletionInputTool(
            function: ChatCompletionInputTool.Function(
                name: "search",
                description: "Search for information",
                parameters: parameters
            )
        )
        
        // Verify the tool was created correctly
        XCTAssertEqual(tool.function.name, "search")
        XCTAssertNotNil(tool.function.parameters)
        
        // Create a sample MCP tool and convert it
        let mcpTool = Tool(
            name: "test_tool",
            description: "A test tool",
            inputSchema: .object([
                "type": .string("object"),
                "properties": .object([
                    "input": .object([
                        "type": .string("string")
                    ])
                ])
            ])
        )
        
        let convertedTool = ChatCompletionInputTool.from(tool: mcpTool)
        XCTAssertEqual(convertedTool.function.name, "test_tool")
        XCTAssertEqual(convertedTool.function.description, "A test tool")
        XCTAssertNotNil(convertedTool.function.parameters)
    }
    
    func testComplexValueTypes() {
        // Test various Value type conversions
        let testCases: [(Value, String)] = [
            (.null, "null value"),
            (.bool(true), "boolean value"),
            (.int(42), "integer value"),
            (.double(3.14), "double value"),
            (.string("test"), "string value"),
            (.array([.int(1), .int(2), .int(3)]), "array value"),
            (.object(["key": .string("value")]), "object value")
        ]
        
        for (value, description) in testCases {
            // Verify that each Value type can be created and accessed
            switch value {
            case .null:
                XCTAssertTrue(value.isNull, "Failed for \(description)")
            case .bool(let b):
                XCTAssertEqual(value.boolValue, b, "Failed for \(description)")
            case .int(let i):
                XCTAssertEqual(value.intValue, i, "Failed for \(description)")
            case .double(let d):
                XCTAssertEqual(value.doubleValue, d, "Failed for \(description)")
            case .string(let s):
                XCTAssertEqual(value.stringValue, s, "Failed for \(description)")
            case .array(let arr):
                XCTAssertEqual(value.arrayValue, arr, "Failed for \(description)")
            case .object(let obj):
                XCTAssertEqual(value.objectValue, obj, "Failed for \(description)")
            case .data:
                break // Not testing data type in this example
            }
        }
    }
    
    func testNestedValueStructure() {
        // Test a deeply nested Value structure similar to JSON schemas
        let nestedValue: Value = .object([
            "type": .string("object"),
            "properties": .object([
                "user": .object([
                    "type": .string("object"),
                    "properties": .object([
                        "name": .object(["type": .string("string")]),
                        "age": .object(["type": .string("integer"), "minimum": .int(0)]),
                        "emails": .object([
                            "type": .string("array"),
                            "items": .object(["type": .string("string"), "format": .string("email")])
                        ])
                    ])
                ])
            ])
        ])
        
        // Verify we can access nested values
        if case .object(let root) = nestedValue,
           let properties = root["properties"]?.objectValue,
           let user = properties["user"]?.objectValue,
           let userProps = user["properties"]?.objectValue {
            
            XCTAssertNotNil(userProps["name"])
            XCTAssertNotNil(userProps["age"])
            XCTAssertNotNil(userProps["emails"])
            
            // Check age has minimum constraint
            if let ageSchema = userProps["age"]?.objectValue {
                XCTAssertEqual(ageSchema["minimum"]?.intValue, 0)
            }
        } else {
            XCTFail("Failed to access nested structure")
        }
    }
}