use gauss_core::message::{Content, Message, Role, Usage};
use gauss_core::tool::{Tool, ToolChoice, ToolParameters};

#[test]
fn test_message_constructors() {
    let sys = Message::system("You are helpful.");
    assert_eq!(sys.role, Role::System);
    assert_eq!(sys.text(), Some("You are helpful."));

    let user = Message::user("Hello!");
    assert_eq!(user.role, Role::User);
    assert_eq!(user.text(), Some("Hello!"));

    let assistant = Message::assistant("Hi there!");
    assert_eq!(assistant.role, Role::Assistant);
    assert_eq!(assistant.text(), Some("Hi there!"));
}

#[test]
fn test_message_tool_calls() {
    let msg = Message {
        role: Role::Assistant,
        content: vec![
            Content::Text {
                text: "Let me check.".to_string(),
            },
            Content::ToolCall {
                id: "tc_1".to_string(),
                name: "get_weather".to_string(),
                arguments: serde_json::json!({"city": "Rome"}),
            },
            Content::ToolCall {
                id: "tc_2".to_string(),
                name: "get_time".to_string(),
                arguments: serde_json::json!({"timezone": "CET"}),
            },
        ],
        name: None,
    };

    let calls = msg.tool_calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].1, "get_weather");
    assert_eq!(calls[1].1, "get_time");
}

#[test]
fn test_message_tool_result() {
    let result = Message::tool_result("tc_1", serde_json::json!({"temp": 25}));
    assert_eq!(result.role, Role::Tool);
    assert!(matches!(
        result.content.first(),
        Some(Content::ToolResult { tool_call_id, .. }) if tool_call_id == "tc_1"
    ));
}

#[test]
fn test_usage_total() {
    let usage = Usage {
        input_tokens: 100,
        output_tokens: 50,
        reasoning_tokens: Some(20),
        ..Default::default()
    };
    assert_eq!(usage.total_tokens(), 150);
}

#[test]
fn test_message_serialization() {
    let msg = Message::user("Hello");
    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"role\":\"user\""));
    assert!(json.contains("Hello"));

    let parsed: Message = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.role, Role::User);
    assert_eq!(parsed.text(), Some("Hello"));
}

#[test]
fn test_tool_builder() {
    let tool = Tool::builder("calculator", "Performs math")
        .parameters_json(serde_json::json!({
            "type": "object",
            "properties": {
                "expression": {"type": "string"}
            },
            "required": ["expression"]
        }))
        .build();

    assert_eq!(tool.name, "calculator");
    assert_eq!(tool.description, "Performs math");
    assert!(!tool.has_execute());
}

#[tokio::test]
async fn test_tool_with_execute() {
    let tool = Tool::builder("echo", "Echoes input")
        .execute(|args| async move { Ok(serde_json::json!({"echoed": args})) })
        .build();

    assert!(tool.has_execute());

    let result = tool
        .execute(serde_json::json!({"message": "hello"}))
        .await
        .unwrap();
    assert!(result["echoed"]["message"].as_str() == Some("hello"));
}

#[tokio::test]
async fn test_tool_without_execute_errors() {
    let tool = Tool::builder("no_exec", "No execute").build();
    let result = tool.execute(serde_json::json!({})).await;
    assert!(result.is_err());
}

#[test]
fn test_tool_choice_default() {
    let tc = ToolChoice::default();
    assert!(matches!(tc, ToolChoice::Auto));
}

#[test]
fn test_tool_choice_serialization() {
    let tc = ToolChoice::Specific {
        name: "calc".to_string(),
    };
    let json = serde_json::to_string(&tc).unwrap();
    assert!(json.contains("calc"));
}
