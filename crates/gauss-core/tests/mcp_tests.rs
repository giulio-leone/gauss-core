use gauss_core::mcp::*;
use gauss_core::tool::Tool;

#[test]
fn test_mcp_tool_creation() {
    let tool = McpTool {
        name: "search".into(),
        description: Some("Search the web".into()),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "query": { "type": "string" }
            },
            "required": ["query"]
        }),
    };

    assert_eq!(tool.name, "search");
    assert_eq!(tool.description.as_deref(), Some("Search the web"));
}

#[test]
fn test_mcp_tool_to_gauss_conversion() {
    let mcp_tool = McpTool {
        name: "calculator".into(),
        description: Some("Perform math".into()),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "expression": { "type": "string", "description": "Math expression" }
            },
            "required": ["expression"]
        }),
    };

    let gauss_tool = mcp_tool_to_gauss(&mcp_tool);
    assert_eq!(gauss_tool.name, "calculator");
    assert_eq!(gauss_tool.description, "Perform math");
}

#[test]
fn test_gauss_tool_to_mcp_conversion() {
    let tool = Tool::builder("greet", "Greet someone").build();
    let mcp_tool = gauss_tool_to_mcp(&tool);
    assert_eq!(mcp_tool.name, "greet");
    assert_eq!(mcp_tool.description.as_deref(), Some("Greet someone"));
}

#[test]
fn test_json_rpc_message_request() {
    let msg = JsonRpcMessage::request(1, "tools/list", serde_json::json!(null));
    assert_eq!(msg.method, Some("tools/list".into()));
    assert_eq!(msg.id, Some(serde_json::json!(1)));
}

#[test]
fn test_json_rpc_message_response() {
    let msg = JsonRpcMessage::response(serde_json::json!(1), serde_json::json!({"tools": []}));
    assert!(msg.result.is_some());
    assert!(msg.error.is_none());
}

#[test]
fn test_json_rpc_error_response() {
    let msg = JsonRpcMessage::error_response(serde_json::json!(1), -32600, "Invalid request");
    assert!(msg.error.is_some());
    let err = msg.error.unwrap();
    assert_eq!(err.code, -32600);
    assert_eq!(err.message, "Invalid request");
}

#[tokio::test]
async fn test_mcp_server_list_tools() {
    let tool = Tool::builder("test_tool", "A test tool").build();
    let mut server = McpServer::new("test-server", "1.0.0");
    server.add_tool(tool);

    let request = JsonRpcMessage::request(1, "tools/list", serde_json::json!(null));
    let response = server.handle_message(request).await.unwrap();

    let tools = response.result.unwrap();
    let tool_list = tools["tools"].as_array().unwrap();
    assert_eq!(tool_list.len(), 1);
    assert_eq!(tool_list[0]["name"], "test_tool");
}

#[tokio::test]
async fn test_mcp_server_unknown_method() {
    let server = McpServer::new("test-server", "1.0.0");
    let request = JsonRpcMessage::request(1, "unknown/method", serde_json::json!(null));
    let response = server.handle_message(request).await.unwrap();

    assert!(response.error.is_some());
    assert_eq!(response.error.unwrap().code, -32601);
}

#[tokio::test]
async fn test_mcp_server_initialize() {
    let mut server = McpServer::new("gauss-mcp", "0.1.0");
    server.add_tool(Tool::builder("echo", "Echo input").build());

    let request = JsonRpcMessage::request(1, "initialize", serde_json::json!({}));
    let response = server.handle_message(request).await.unwrap();

    let result = response.result.unwrap();
    assert_eq!(result["serverInfo"]["name"], "gauss-mcp");
    assert_eq!(result["protocolVersion"], "2024-11-05");
    assert!(result["capabilities"]["tools"].is_object());
}

#[tokio::test]
async fn test_mcp_server_tools_call_unknown() {
    let server = McpServer::new("test", "1.0");
    let request = JsonRpcMessage::request(
        1,
        "tools/call",
        serde_json::json!({"name": "nonexistent", "arguments": {}}),
    );
    let response = server.handle_message(request).await.unwrap();
    assert!(response.error.is_some());
    assert_eq!(response.error.unwrap().code, -32602);
}

#[tokio::test]
async fn test_mcp_server_capabilities() {
    let server = McpServer::new("test", "1.0");
    let caps = server.capabilities();
    assert!(caps.tools.is_some());
    assert!(caps.resources.is_none());

    let mut server2 = McpServer::new("test", "1.0");
    server2.add_resource(McpResource {
        uri: "file://test.txt".into(),
        name: "test".into(),
        description: None,
        mime_type: None,
    });
    let caps2 = server2.capabilities();
    assert!(caps2.resources.is_some());
}

#[tokio::test]
async fn test_transport_client_with_server() {
    use std::sync::Arc;
    use tokio::sync::Mutex;

    /// In-memory transport for testing: two channels forming a bidirectional pipe.
    struct ChannelTransport {
        tx: tokio::sync::mpsc::Sender<JsonRpcMessage>,
        rx: Arc<Mutex<tokio::sync::mpsc::Receiver<JsonRpcMessage>>>,
    }

    #[async_trait::async_trait]
    impl McpTransport for ChannelTransport {
        async fn send(&self, message: &JsonRpcMessage) -> gauss_core::error::Result<()> {
            self.tx
                .send(message.clone())
                .await
                .map_err(|e| gauss_core::error::GaussError::tool("mcp", format!("{e}")))?;
            Ok(())
        }
        async fn receive(&self) -> gauss_core::error::Result<JsonRpcMessage> {
            self.rx
                .lock()
                .await
                .recv()
                .await
                .ok_or_else(|| gauss_core::error::GaussError::tool("mcp", "Channel closed"))
        }
        async fn close(&self) -> gauss_core::error::Result<()> {
            Ok(())
        }
    }

    fn channel_pair() -> (ChannelTransport, ChannelTransport) {
        let (tx1, rx1) = tokio::sync::mpsc::channel(32);
        let (tx2, rx2) = tokio::sync::mpsc::channel(32);
        (
            ChannelTransport {
                tx: tx1,
                rx: Arc::new(Mutex::new(rx2)),
            },
            ChannelTransport {
                tx: tx2,
                rx: Arc::new(Mutex::new(rx1)),
            },
        )
    }

    // Create server with a tool
    let mut server = McpServer::new("test-server", "1.0.0");
    server.add_tool(Tool::builder("ping", "Returns pong").build());

    // Create transport pair
    let (client_transport, server_transport) = channel_pair();

    // Spawn server in background (process one request then stop)
    let server_handle = tokio::spawn(async move {
        let msg = server_transport.receive().await.unwrap();
        let resp = server.handle_message(msg).await.unwrap();
        server_transport.send(&resp).await.unwrap();
    });

    // Client calls list_tools
    let mut client = TransportMcpClient::new(client_transport);
    let tools = client.list_tools().await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "ping");

    server_handle.await.unwrap();
}
