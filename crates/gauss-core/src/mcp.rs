//! MCP (Model Context Protocol) — client/server for tool interop.
//!
//! Implements the MCP specification for connecting to external tool servers
//! and exposing Gauss tools as MCP endpoints.

use crate::error;
use crate::tool::Tool;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// MCP Types (per spec)
// ---------------------------------------------------------------------------

/// An MCP tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(rename = "inputSchema")]
    pub input_schema: serde_json::Value,
}

/// An MCP resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResource {
    pub uri: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

/// An MCP prompt template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpPrompt {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub arguments: Vec<McpPromptArgument>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpPromptArgument {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub required: bool,
}

/// JSON-RPC message envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcMessage {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl JsonRpcMessage {
    pub fn request(id: u64, method: &str, params: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: Some(serde_json::Value::Number(id.into())),
            method: Some(method.to_string()),
            params: Some(params),
            result: None,
            error: None,
        }
    }

    pub fn response(id: serde_json::Value, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: Some(id),
            method: None,
            params: None,
            result: Some(result),
            error: None,
        }
    }

    pub fn error_response(id: serde_json::Value, code: i64, message: &str) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: Some(id),
            method: None,
            params: None,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.to_string(),
                data: None,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Server Capabilities
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct McpServerCapabilities {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<McpToolsCapability>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resources: Option<McpResourcesCapability>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompts: Option<McpPromptsCapability>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct McpToolsCapability {
    #[serde(default)]
    pub list_changed: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct McpResourcesCapability {
    #[serde(default)]
    pub subscribe: bool,
    #[serde(default)]
    pub list_changed: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct McpPromptsCapability {
    #[serde(default)]
    pub list_changed: bool,
}

// ---------------------------------------------------------------------------
// MCP Client Trait
// ---------------------------------------------------------------------------

/// Client for connecting to MCP servers and discovering/calling tools.
#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
pub trait McpClient: Send + Sync {
    /// Initialize the connection and exchange capabilities.
    async fn initialize(&mut self) -> error::Result<McpServerCapabilities>;

    /// List available tools from the server.
    async fn list_tools(&self) -> error::Result<Vec<McpTool>>;

    /// Call a tool on the server.
    async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> error::Result<serde_json::Value>;

    /// List available resources.
    async fn list_resources(&self) -> error::Result<Vec<McpResource>>;

    /// Read a resource by URI.
    async fn read_resource(&self, uri: &str) -> error::Result<serde_json::Value>;

    /// Close the connection.
    async fn close(&mut self) -> error::Result<()>;
}

#[cfg(target_arch = "wasm32")]
#[async_trait(?Send)]
pub trait McpClient {
    async fn initialize(&mut self) -> error::Result<McpServerCapabilities>;
    async fn list_tools(&self) -> error::Result<Vec<McpTool>>;
    async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> error::Result<serde_json::Value>;
    async fn list_resources(&self) -> error::Result<Vec<McpResource>>;
    async fn read_resource(&self, uri: &str) -> error::Result<serde_json::Value>;
    async fn close(&mut self) -> error::Result<()>;
}

// ---------------------------------------------------------------------------
// MCP → Gauss Adapter
// ---------------------------------------------------------------------------

/// Convert an MCP tool into a Gauss Tool (using the builder pattern).
pub fn mcp_tool_to_gauss(mcp_tool: &McpTool) -> Tool {
    let mut params = crate::tool::ToolParameters::default();
    if let Some(obj) = mcp_tool
        .input_schema
        .get("properties")
        .and_then(|p| p.as_object())
    {
        params.properties = Some(obj.clone());
    }
    if let Some(arr) = mcp_tool
        .input_schema
        .get("required")
        .and_then(|r| r.as_array())
    {
        params.required = Some(
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect(),
        );
    }

    Tool::builder(
        &mcp_tool.name,
        mcp_tool.description.as_deref().unwrap_or(""),
    )
    .parameters(params)
    .build()
}

/// Convert a Gauss Tool into an MCP tool definition.
pub fn gauss_tool_to_mcp(tool: &Tool) -> McpTool {
    let mut schema = serde_json::json!({ "type": "object" });
    if let Some(ref props) = tool.parameters.properties {
        schema["properties"] = serde_json::Value::Object(props.clone());
    }
    if let Some(ref req) = tool.parameters.required {
        schema["required"] = serde_json::json!(req);
    }

    McpTool {
        name: tool.name.clone(),
        description: Some(tool.description.clone()),
        input_schema: schema,
    }
}

// ---------------------------------------------------------------------------
// MCP Server
// ---------------------------------------------------------------------------

/// MCP Server that exposes Gauss tools to MCP clients.
pub struct McpServer {
    pub name: String,
    pub version: String,
    tools: Vec<Tool>,
    resources: Vec<McpResource>,
}

impl McpServer {
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            tools: Vec::new(),
            resources: Vec::new(),
        }
    }

    pub fn add_tool(&mut self, tool: Tool) {
        self.tools.push(tool);
    }

    pub fn add_resource(&mut self, resource: McpResource) {
        self.resources.push(resource);
    }

    /// Handle an incoming JSON-RPC message.
    pub async fn handle_message(&self, msg: JsonRpcMessage) -> error::Result<JsonRpcMessage> {
        let id = msg.id.clone().unwrap_or(serde_json::Value::Null);
        let method = msg.method.as_deref().unwrap_or("");

        match method {
            "initialize" => {
                let caps = McpServerCapabilities {
                    tools: Some(McpToolsCapability::default()),
                    resources: if self.resources.is_empty() {
                        None
                    } else {
                        Some(McpResourcesCapability::default())
                    },
                    prompts: None,
                };
                Ok(JsonRpcMessage::response(
                    id,
                    serde_json::json!({
                        "protocolVersion": "2024-11-05",
                        "capabilities": caps,
                        "serverInfo": {
                            "name": self.name,
                            "version": self.version,
                        }
                    }),
                ))
            }
            "tools/list" => {
                let tools: Vec<McpTool> = self.tools.iter().map(gauss_tool_to_mcp).collect();
                Ok(JsonRpcMessage::response(
                    id,
                    serde_json::json!({ "tools": tools }),
                ))
            }
            "tools/call" => {
                let params = msg.params.unwrap_or(serde_json::Value::Null);
                let tool_name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
                let args = params
                    .get("arguments")
                    .cloned()
                    .unwrap_or(serde_json::Value::Object(Default::default()));

                let tool = self.tools.iter().find(|t| t.name == tool_name);

                match tool {
                    Some(t) => {
                        if t.has_execute() {
                            let result: serde_json::Value = t.execute(args).await?;
                            Ok(JsonRpcMessage::response(
                                id,
                                serde_json::json!({
                                    "content": [{ "type": "text", "text": result.to_string() }]
                                }),
                            ))
                        } else {
                            Ok(JsonRpcMessage::error_response(
                                id,
                                -32603,
                                "Tool has no execute function",
                            ))
                        }
                    }
                    None => Ok(JsonRpcMessage::error_response(
                        id,
                        -32602,
                        &format!("Unknown tool: {tool_name}"),
                    )),
                }
            }
            "resources/list" => Ok(JsonRpcMessage::response(
                id,
                serde_json::json!({ "resources": self.resources }),
            )),
            _ => Ok(JsonRpcMessage::error_response(
                id,
                -32601,
                &format!("Method not found: {method}"),
            )),
        }
    }

    /// Get server capabilities.
    pub fn capabilities(&self) -> McpServerCapabilities {
        McpServerCapabilities {
            tools: Some(McpToolsCapability::default()),
            resources: if self.resources.is_empty() {
                None
            } else {
                Some(McpResourcesCapability::default())
            },
            prompts: None,
        }
    }
}

// ---------------------------------------------------------------------------
// MCP Transport Trait
// ---------------------------------------------------------------------------

/// Transport layer for MCP communication.
#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
pub trait McpTransport: Send + Sync {
    /// Send a JSON-RPC message.
    async fn send(&self, message: &JsonRpcMessage) -> error::Result<()>;
    /// Receive the next JSON-RPC message.
    async fn receive(&self) -> error::Result<JsonRpcMessage>;
    /// Close the transport.
    async fn close(&self) -> error::Result<()>;
}

// ---------------------------------------------------------------------------
// Stdio Transport (native only)
// ---------------------------------------------------------------------------

/// stdio-based MCP transport using newline-delimited JSON.
#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
pub struct StdioTransport {
    reader: tokio::sync::Mutex<tokio::io::BufReader<tokio::io::Stdin>>,
    writer: tokio::sync::Mutex<tokio::io::Stdout>,
}

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
impl StdioTransport {
    pub fn new() -> Self {
        Self {
            reader: tokio::sync::Mutex::new(tokio::io::BufReader::new(tokio::io::stdin())),
            writer: tokio::sync::Mutex::new(tokio::io::stdout()),
        }
    }
}

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
impl Default for StdioTransport {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
#[async_trait]
impl McpTransport for StdioTransport {
    async fn send(&self, message: &JsonRpcMessage) -> error::Result<()> {
        use tokio::io::AsyncWriteExt;
        let json = serde_json::to_string(message)
            .map_err(|e| error::GaussError::tool("mcp", format!("Serialize error: {e}")))?;
        let mut writer = self.writer.lock().await;
        writer
            .write_all(json.as_bytes())
            .await
            .map_err(|e| error::GaussError::tool("mcp", format!("Write error: {e}")))?;
        writer
            .write_all(b"\n")
            .await
            .map_err(|e| error::GaussError::tool("mcp", format!("Write newline error: {e}")))?;
        writer
            .flush()
            .await
            .map_err(|e| error::GaussError::tool("mcp", format!("Flush error: {e}")))?;
        Ok(())
    }

    async fn receive(&self) -> error::Result<JsonRpcMessage> {
        use tokio::io::AsyncBufReadExt;
        let mut reader = self.reader.lock().await;
        let mut line = String::new();
        reader
            .read_line(&mut line)
            .await
            .map_err(|e| error::GaussError::tool("mcp", format!("Read error: {e}")))?;
        if line.is_empty() {
            return Err(error::GaussError::tool("mcp", "EOF on stdin"));
        }
        serde_json::from_str(line.trim())
            .map_err(|e| error::GaussError::tool("mcp", format!("Parse error: {e}")))
    }

    async fn close(&self) -> error::Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Child Process Stdio Transport (for MCP clients connecting to servers)
// ---------------------------------------------------------------------------

/// Transport that communicates with an MCP server running as a child process.
#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
pub struct ChildProcessTransport {
    stdin: tokio::sync::Mutex<tokio::process::ChildStdin>,
    reader: tokio::sync::Mutex<tokio::io::BufReader<tokio::process::ChildStdout>>,
    child: tokio::sync::Mutex<tokio::process::Child>,
}

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
impl ChildProcessTransport {
    /// Spawn a child process and create a transport for it.
    pub fn spawn(command: &str, args: &[&str]) -> error::Result<Self> {
        use std::process::Stdio;
        let mut child = tokio::process::Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| error::GaussError::tool("mcp", format!("Spawn error: {e}")))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| error::GaussError::tool("mcp", "No stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| error::GaussError::tool("mcp", "No stdout"))?;

        Ok(Self {
            stdin: tokio::sync::Mutex::new(stdin),
            reader: tokio::sync::Mutex::new(tokio::io::BufReader::new(stdout)),
            child: tokio::sync::Mutex::new(child),
        })
    }
}

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
#[async_trait]
impl McpTransport for ChildProcessTransport {
    async fn send(&self, message: &JsonRpcMessage) -> error::Result<()> {
        use tokio::io::AsyncWriteExt;
        let json = serde_json::to_string(message)
            .map_err(|e| error::GaussError::tool("mcp", format!("Serialize error: {e}")))?;
        let mut stdin = self.stdin.lock().await;
        stdin
            .write_all(json.as_bytes())
            .await
            .map_err(|e| error::GaussError::tool("mcp", format!("Write error: {e}")))?;
        stdin
            .write_all(b"\n")
            .await
            .map_err(|e| error::GaussError::tool("mcp", format!("Write newline error: {e}")))?;
        stdin
            .flush()
            .await
            .map_err(|e| error::GaussError::tool("mcp", format!("Flush error: {e}")))?;
        Ok(())
    }

    async fn receive(&self) -> error::Result<JsonRpcMessage> {
        use tokio::io::AsyncBufReadExt;
        let mut reader = self.reader.lock().await;
        let mut line = String::new();
        reader
            .read_line(&mut line)
            .await
            .map_err(|e| error::GaussError::tool("mcp", format!("Read error: {e}")))?;
        if line.is_empty() {
            return Err(error::GaussError::tool("mcp", "EOF on child stdout"));
        }
        serde_json::from_str(line.trim())
            .map_err(|e| error::GaussError::tool("mcp", format!("Parse error: {e}")))
    }

    async fn close(&self) -> error::Result<()> {
        let mut child = self.child.lock().await;
        let _ = child.kill().await;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Stdio MCP Client (connects to an MCP server via stdio transport)
// ---------------------------------------------------------------------------

/// An MCP client that communicates over a transport.
#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
pub struct TransportMcpClient<T: McpTransport> {
    transport: T,
    next_id: std::sync::atomic::AtomicU64,
}

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
impl<T: McpTransport> TransportMcpClient<T> {
    pub fn new(transport: T) -> Self {
        Self {
            transport,
            next_id: std::sync::atomic::AtomicU64::new(1),
        }
    }

    fn next_id(&self) -> u64 {
        self.next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    async fn request(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> error::Result<JsonRpcMessage> {
        let msg = JsonRpcMessage::request(self.next_id(), method, params);
        self.transport.send(&msg).await?;
        self.transport.receive().await
    }
}

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
#[async_trait]
impl<T: McpTransport> McpClient for TransportMcpClient<T> {
    async fn initialize(&mut self) -> error::Result<McpServerCapabilities> {
        let resp = self
            .request(
                "initialize",
                serde_json::json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": { "name": "gauss", "version": env!("CARGO_PKG_VERSION") }
                }),
            )
            .await?;

        if let Some(err) = resp.error {
            return Err(error::GaussError::tool(
                "mcp",
                format!("Initialize error: {}", err.message),
            ));
        }

        let result = resp.result.unwrap_or_default();
        let caps = result
            .get("capabilities")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        serde_json::from_value(caps)
            .map_err(|e| error::GaussError::tool("mcp", format!("Parse caps: {e}")))
    }

    async fn list_tools(&self) -> error::Result<Vec<McpTool>> {
        let resp = self.request("tools/list", serde_json::json!({})).await?;
        if let Some(err) = resp.error {
            return Err(error::GaussError::tool(
                "mcp",
                format!("list_tools error: {}", err.message),
            ));
        }
        let result = resp.result.unwrap_or_default();
        let tools = result
            .get("tools")
            .cloned()
            .unwrap_or(serde_json::Value::Array(vec![]));
        serde_json::from_value(tools)
            .map_err(|e| error::GaussError::tool("mcp", format!("Parse tools: {e}")))
    }

    async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> error::Result<serde_json::Value> {
        let resp = self
            .request(
                "tools/call",
                serde_json::json!({ "name": name, "arguments": arguments }),
            )
            .await?;
        if let Some(err) = resp.error {
            return Err(error::GaussError::tool(
                "mcp",
                format!("call_tool error: {}", err.message),
            ));
        }
        Ok(resp.result.unwrap_or_default())
    }

    async fn list_resources(&self) -> error::Result<Vec<McpResource>> {
        let resp = self
            .request("resources/list", serde_json::json!({}))
            .await?;
        if let Some(err) = resp.error {
            return Err(error::GaussError::tool(
                "mcp",
                format!("list_resources error: {}", err.message),
            ));
        }
        let result = resp.result.unwrap_or_default();
        let resources = result
            .get("resources")
            .cloned()
            .unwrap_or(serde_json::Value::Array(vec![]));
        serde_json::from_value(resources)
            .map_err(|e| error::GaussError::tool("mcp", format!("Parse resources: {e}")))
    }

    async fn read_resource(&self, uri: &str) -> error::Result<serde_json::Value> {
        let resp = self
            .request("resources/read", serde_json::json!({ "uri": uri }))
            .await?;
        if let Some(err) = resp.error {
            return Err(error::GaussError::tool(
                "mcp",
                format!("read_resource error: {}", err.message),
            ));
        }
        Ok(resp.result.unwrap_or_default())
    }

    async fn close(&mut self) -> error::Result<()> {
        self.transport.close().await
    }
}

// ---------------------------------------------------------------------------
// Server Runner (serves McpServer over a transport)
// ---------------------------------------------------------------------------

/// Run an MCP server over a transport, processing messages until EOF/error.
#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
pub async fn serve<T: McpTransport>(server: &McpServer, transport: &T) -> error::Result<()> {
    loop {
        let msg = match transport.receive().await {
            Ok(m) => m,
            Err(_) => break, // EOF or transport closed
        };
        let resp = server.handle_message(msg).await?;
        transport.send(&resp).await?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// HTTP/SSE Transport (native only)
// ---------------------------------------------------------------------------

/// HTTP-based MCP transport. Sends JSON-RPC as POST, receives responses.
/// Also supports Server-Sent Events (SSE) for streaming server notifications.
#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
pub struct HttpTransport {
    client: reqwest::Client,
    endpoint: String,
    pending: tokio::sync::Mutex<Vec<JsonRpcMessage>>,
}

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
impl HttpTransport {
    /// Create a new HTTP transport pointing at the MCP server endpoint.
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            endpoint: endpoint.into(),
            pending: tokio::sync::Mutex::new(Vec::new()),
        }
    }

    /// Create with custom reqwest client (for auth headers, timeouts, etc.).
    pub fn with_client(client: reqwest::Client, endpoint: impl Into<String>) -> Self {
        Self {
            client,
            endpoint: endpoint.into(),
            pending: tokio::sync::Mutex::new(Vec::new()),
        }
    }

    /// Connect to an SSE endpoint and collect events into pending messages.
    pub async fn connect_sse(&self, sse_url: &str) -> error::Result<()> {
        let resp = self
            .client
            .get(sse_url)
            .header("Accept", "text/event-stream")
            .send()
            .await
            .map_err(|e| error::GaussError::tool("mcp", format!("SSE connect error: {e}")))?;

        if !resp.status().is_success() {
            return Err(error::GaussError::tool(
                "mcp",
                format!("SSE HTTP {}", resp.status()),
            ));
        }

        let text = resp
            .text()
            .await
            .map_err(|e| error::GaussError::tool("mcp", format!("SSE read error: {e}")))?;

        // Parse SSE events (data: lines)
        let mut pending = self.pending.lock().await;
        for line in text.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                if let Ok(msg) = serde_json::from_str::<JsonRpcMessage>(data) {
                    pending.push(msg);
                }
            }
        }
        Ok(())
    }
}

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
#[async_trait]
impl McpTransport for HttpTransport {
    async fn send(&self, message: &JsonRpcMessage) -> error::Result<()> {
        let resp = self
            .client
            .post(&self.endpoint)
            .header("Content-Type", "application/json")
            .json(message)
            .send()
            .await
            .map_err(|e| error::GaussError::tool("mcp", format!("HTTP send error: {e}")))?;

        if !resp.status().is_success() {
            return Err(error::GaussError::tool(
                "mcp",
                format!("HTTP error: {}", resp.status()),
            ));
        }

        // If the response body contains a JSON-RPC response, queue it
        let body = resp
            .text()
            .await
            .map_err(|e| error::GaussError::tool("mcp", format!("HTTP read error: {e}")))?;

        if !body.trim().is_empty() {
            if let Ok(msg) = serde_json::from_str::<JsonRpcMessage>(&body) {
                self.pending.lock().await.push(msg);
            }
        }
        Ok(())
    }

    async fn receive(&self) -> error::Result<JsonRpcMessage> {
        // Return from pending queue
        let mut pending = self.pending.lock().await;
        if pending.is_empty() {
            return Err(error::GaussError::tool(
                "mcp",
                "No pending messages (call send first or connect_sse)",
            ));
        }
        Ok(pending.remove(0))
    }

    async fn close(&self) -> error::Result<()> {
        Ok(())
    }
}
