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

/// Result of getting a prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpPromptResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub messages: Vec<McpPromptMessage>,
}

/// A message in a prompt result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpPromptMessage {
    pub role: String,
    pub content: McpContent,
}

/// Content in an MCP message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum McpContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { data: String, mime_type: String },
    #[serde(rename = "resource")]
    Resource { resource: McpResourceContent },
}

/// Embedded resource content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResourceContent {
    pub uri: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blob: Option<String>,
}

/// MCP sampling request (client → server model invocation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpSamplingRequest {
    pub messages: Vec<McpSamplingMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_preferences: Option<McpModelPreferences>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    #[serde(rename = "includeContext", skip_serializing_if = "Option::is_none")]
    pub include_context: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(rename = "maxTokens")]
    pub max_tokens: u32,
    #[serde(rename = "stopSequences", skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// A message in a sampling request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpSamplingMessage {
    pub role: String,
    pub content: McpContent,
}

/// Model preferences for sampling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpModelPreferences {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hints: Option<Vec<McpModelHint>>,
    #[serde(rename = "costPriority", skip_serializing_if = "Option::is_none")]
    pub cost_priority: Option<f64>,
    #[serde(rename = "speedPriority", skip_serializing_if = "Option::is_none")]
    pub speed_priority: Option<f64>,
    #[serde(rename = "intelligencePriority", skip_serializing_if = "Option::is_none")]
    pub intelligence_priority: Option<f64>,
}

/// A hint for model selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpModelHint {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// MCP sampling response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpSamplingResponse {
    pub role: String,
    pub content: McpContent,
    pub model: String,
    #[serde(rename = "stopReason", skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
}

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

    /// List available prompts.
    async fn list_prompts(&self) -> error::Result<Vec<McpPrompt>>;

    /// Get a prompt by name with arguments.
    async fn get_prompt(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> error::Result<McpPromptResult>;

    /// Send a sampling request to the server.
    async fn create_message(
        &self,
        request: McpSamplingRequest,
    ) -> error::Result<McpSamplingResponse>;

    /// Check if the server is healthy and responsive.
    async fn ping(&self) -> error::Result<()>;

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
    async fn list_prompts(&self) -> error::Result<Vec<McpPrompt>>;
    async fn get_prompt(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> error::Result<McpPromptResult>;
    async fn create_message(
        &self,
        request: McpSamplingRequest,
    ) -> error::Result<McpSamplingResponse>;
    async fn ping(&self) -> error::Result<()>;
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
    prompts: Vec<McpPrompt>,
}

impl McpServer {
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            tools: Vec::new(),
            resources: Vec::new(),
            prompts: Vec::new(),
        }
    }

    pub fn add_tool(&mut self, tool: Tool) {
        self.tools.push(tool);
    }

    pub fn add_resource(&mut self, resource: McpResource) {
        self.resources.push(resource);
    }

    pub fn add_prompt(&mut self, prompt: McpPrompt) {
        self.prompts.push(prompt);
    }

    /// Handle an incoming JSON-RPC message.
    pub async fn handle_message(&self, msg: JsonRpcMessage) -> error::Result<JsonRpcMessage> {
        let id = msg.id.clone().unwrap_or(serde_json::Value::Null);
        let method = msg.method.as_deref().unwrap_or("");

        match method {
            "initialize" => {
                let caps = self.capabilities();
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
            "prompts/list" => Ok(JsonRpcMessage::response(
                id,
                serde_json::json!({ "prompts": self.prompts }),
            )),
            "prompts/get" => {
                let params = msg.params.unwrap_or(serde_json::Value::Null);
                let name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
                match self.prompts.iter().find(|p| p.name == name) {
                    Some(_prompt) => {
                        // Return a basic prompt result — users customize via prompt handlers
                        Ok(JsonRpcMessage::response(
                            id,
                            serde_json::json!({
                                "description": _prompt.description,
                                "messages": []
                            }),
                        ))
                    }
                    None => Ok(JsonRpcMessage::error_response(
                        id,
                        -32602,
                        &format!("Unknown prompt: {name}"),
                    )),
                }
            }
            "ping" => Ok(JsonRpcMessage::response(id, serde_json::json!({}))),
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
            prompts: if self.prompts.is_empty() {
                None
            } else {
                Some(McpPromptsCapability::default())
            },
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

    async fn list_prompts(&self) -> error::Result<Vec<McpPrompt>> {
        let resp = self
            .request("prompts/list", serde_json::json!({}))
            .await?;
        if let Some(err) = resp.error {
            return Err(error::GaussError::tool(
                "mcp",
                format!("list_prompts error: {}", err.message),
            ));
        }
        let result = resp.result.unwrap_or_default();
        let prompts = result
            .get("prompts")
            .cloned()
            .unwrap_or(serde_json::Value::Array(vec![]));
        serde_json::from_value(prompts)
            .map_err(|e| error::GaussError::tool("mcp", format!("Parse prompts: {e}")))
    }

    async fn get_prompt(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> error::Result<McpPromptResult> {
        let resp = self
            .request(
                "prompts/get",
                serde_json::json!({ "name": name, "arguments": arguments }),
            )
            .await?;
        if let Some(err) = resp.error {
            return Err(error::GaussError::tool(
                "mcp",
                format!("get_prompt error: {}", err.message),
            ));
        }
        let result = resp.result.unwrap_or_default();
        serde_json::from_value(result)
            .map_err(|e| error::GaussError::tool("mcp", format!("Parse prompt result: {e}")))
    }

    async fn create_message(
        &self,
        request: McpSamplingRequest,
    ) -> error::Result<McpSamplingResponse> {
        let params = serde_json::to_value(&request)
            .map_err(|e| error::GaussError::tool("mcp", format!("Serialize sampling: {e}")))?;
        let resp = self.request("sampling/createMessage", params).await?;
        if let Some(err) = resp.error {
            return Err(error::GaussError::tool(
                "mcp",
                format!("create_message error: {}", err.message),
            ));
        }
        let result = resp.result.unwrap_or_default();
        serde_json::from_value(result)
            .map_err(|e| error::GaussError::tool("mcp", format!("Parse sampling response: {e}")))
    }

    async fn ping(&self) -> error::Result<()> {
        let resp = self.request("ping", serde_json::json!({})).await?;
        if let Some(err) = resp.error {
            return Err(error::GaussError::tool(
                "mcp",
                format!("ping error: {}", err.message),
            ));
        }
        Ok(())
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
            if let Some(data) = line.strip_prefix("data: ")
                && let Ok(msg) = serde_json::from_str::<JsonRpcMessage>(data)
            {
                pending.push(msg);
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

        if !body.trim().is_empty()
            && let Ok(msg) = serde_json::from_str::<JsonRpcMessage>(&body)
        {
            self.pending.lock().await.push(msg);
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcp_tool_serde() {
        let tool = McpTool {
            name: "test".into(),
            description: Some("A test tool".into()),
            input_schema: serde_json::json!({"type": "object"}),
        };
        let json = serde_json::to_string(&tool).unwrap();
        let parsed: McpTool = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "test");
        assert_eq!(parsed.description.as_deref(), Some("A test tool"));
    }

    #[test]
    fn test_mcp_resource_serde() {
        let res = McpResource {
            uri: "file:///test.txt".into(),
            name: "test.txt".into(),
            description: Some("A test file".into()),
            mime_type: Some("text/plain".into()),
        };
        let json = serde_json::to_string(&res).unwrap();
        let parsed: McpResource = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.uri, "file:///test.txt");
    }

    #[test]
    fn test_mcp_prompt_serde() {
        let prompt = McpPrompt {
            name: "summarize".into(),
            description: Some("Summarize text".into()),
            arguments: vec![McpPromptArgument {
                name: "text".into(),
                description: Some("Text to summarize".into()),
                required: true,
            }],
        };
        let json = serde_json::to_string(&prompt).unwrap();
        let parsed: McpPrompt = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "summarize");
        assert_eq!(parsed.arguments.len(), 1);
        assert_eq!(parsed.arguments[0].required, true);
    }

    #[test]
    fn test_json_rpc_request() {
        let msg = JsonRpcMessage::request(1, "tools/list", serde_json::json!({}));
        assert_eq!(msg.method.as_deref(), Some("tools/list"));
        assert_eq!(msg.id, Some(serde_json::json!(1)));
    }

    #[test]
    fn test_json_rpc_response() {
        let msg = JsonRpcMessage::response(
            serde_json::json!(1),
            serde_json::json!({"tools": []}),
        );
        assert!(msg.result.is_some());
        assert!(msg.error.is_none());
    }

    #[test]
    fn test_json_rpc_error_response() {
        let msg = JsonRpcMessage::error_response(serde_json::json!(1), -32601, "Not found");
        assert!(msg.error.is_some());
        assert_eq!(msg.error.as_ref().unwrap().code, -32601);
        assert_eq!(msg.error.as_ref().unwrap().message, "Not found");
    }

    #[test]
    fn test_server_capabilities_default() {
        let caps = McpServerCapabilities::default();
        assert!(caps.tools.is_none());
        assert!(caps.resources.is_none());
        assert!(caps.prompts.is_none());
    }

    #[test]
    fn test_mcp_content_text() {
        let content = McpContent::Text { text: "hello".into() };
        let json = serde_json::to_string(&content).unwrap();
        assert!(json.contains("\"type\":\"text\""));
        assert!(json.contains("hello"));
    }

    #[test]
    fn test_mcp_content_image() {
        let content = McpContent::Image {
            data: "base64data".into(),
            mime_type: "image/png".into(),
        };
        let json = serde_json::to_string(&content).unwrap();
        assert!(json.contains("\"type\":\"image\""));
    }

    #[test]
    fn test_sampling_request_serde() {
        let req = McpSamplingRequest {
            messages: vec![McpSamplingMessage {
                role: "user".into(),
                content: McpContent::Text { text: "Hello".into() },
            }],
            model_preferences: None,
            system_prompt: Some("You are helpful".into()),
            include_context: None,
            temperature: Some(0.7),
            max_tokens: 1000,
            stop_sequences: None,
            metadata: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: McpSamplingRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.messages.len(), 1);
        assert_eq!(parsed.max_tokens, 1000);
        assert_eq!(parsed.temperature, Some(0.7));
    }

    #[test]
    fn test_sampling_response_serde() {
        let resp = McpSamplingResponse {
            role: "assistant".into(),
            content: McpContent::Text { text: "Response".into() },
            model: "gpt-4".into(),
            stop_reason: Some("end_turn".into()),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: McpSamplingResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.model, "gpt-4");
        assert_eq!(parsed.stop_reason.as_deref(), Some("end_turn"));
    }

    #[test]
    fn test_prompt_result_serde() {
        let result = McpPromptResult {
            description: Some("Test prompt".into()),
            messages: vec![McpPromptMessage {
                role: "user".into(),
                content: McpContent::Text { text: "Hello".into() },
            }],
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: McpPromptResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.messages.len(), 1);
    }

    #[test]
    fn test_model_preferences_serde() {
        let prefs = McpModelPreferences {
            hints: Some(vec![McpModelHint {
                name: Some("claude-3".into()),
            }]),
            cost_priority: Some(0.3),
            speed_priority: Some(0.5),
            intelligence_priority: Some(0.8),
        };
        let json = serde_json::to_string(&prefs).unwrap();
        assert!(json.contains("costPriority"));
        assert!(json.contains("speedPriority"));
        assert!(json.contains("intelligencePriority"));
    }

    #[test]
    fn test_mcp_tool_to_gauss_roundtrip() {
        let mcp_tool = McpTool {
            name: "calculator".into(),
            description: Some("Does math".into()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "expression": { "type": "string" }
                },
                "required": ["expression"]
            }),
        };
        let gauss_tool = mcp_tool_to_gauss(&mcp_tool);
        assert_eq!(gauss_tool.name, "calculator");
        assert_eq!(gauss_tool.description, "Does math");

        let back = gauss_tool_to_mcp(&gauss_tool);
        assert_eq!(back.name, "calculator");
    }

    #[test]
    fn test_server_add_prompt() {
        let mut server = McpServer::new("test", "1.0");
        assert!(server.prompts.is_empty());
        server.add_prompt(McpPrompt {
            name: "greet".into(),
            description: Some("A greeting".into()),
            arguments: vec![],
        });
        assert_eq!(server.prompts.len(), 1);
        let caps = server.capabilities();
        assert!(caps.prompts.is_some());
    }

    #[test]
    fn test_server_capabilities_with_all() {
        let mut server = McpServer::new("test", "1.0");
        server.add_tool(Tool::builder("t1", "Test tool").build());
        server.add_resource(McpResource {
            uri: "test://res".into(),
            name: "res".into(),
            description: None,
            mime_type: None,
        });
        server.add_prompt(McpPrompt {
            name: "p1".into(),
            description: None,
            arguments: vec![],
        });
        let caps = server.capabilities();
        assert!(caps.tools.is_some());
        assert!(caps.resources.is_some());
        assert!(caps.prompts.is_some());
    }

    #[tokio::test]
    async fn test_server_handle_initialize() {
        let server = McpServer::new("test", "1.0");
        let msg = JsonRpcMessage::request(1, "initialize", serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": { "name": "test", "version": "1.0" }
        }));
        let resp = server.handle_message(msg).await.unwrap();
        assert!(resp.result.is_some());
        let result = resp.result.unwrap();
        assert_eq!(result["protocolVersion"], "2024-11-05");
    }

    #[tokio::test]
    async fn test_server_handle_tools_list() {
        let mut server = McpServer::new("test", "1.0");
        server.add_tool(Tool::builder("calc", "Calculator").build());
        let msg = JsonRpcMessage::request(2, "tools/list", serde_json::json!({}));
        let resp = server.handle_message(msg).await.unwrap();
        let tools = resp.result.unwrap()["tools"].as_array().unwrap().len();
        assert_eq!(tools, 1);
    }

    #[tokio::test]
    async fn test_server_handle_prompts_list() {
        let mut server = McpServer::new("test", "1.0");
        server.add_prompt(McpPrompt {
            name: "greet".into(),
            description: Some("Greet user".into()),
            arguments: vec![],
        });
        let msg = JsonRpcMessage::request(3, "prompts/list", serde_json::json!({}));
        let resp = server.handle_message(msg).await.unwrap();
        let prompts = resp.result.unwrap()["prompts"].as_array().unwrap().len();
        assert_eq!(prompts, 1);
    }

    #[tokio::test]
    async fn test_server_handle_prompts_get() {
        let mut server = McpServer::new("test", "1.0");
        server.add_prompt(McpPrompt {
            name: "greet".into(),
            description: Some("Greet user".into()),
            arguments: vec![],
        });
        let msg = JsonRpcMessage::request(4, "prompts/get", serde_json::json!({"name": "greet"}));
        let resp = server.handle_message(msg).await.unwrap();
        assert!(resp.result.is_some());
    }

    #[tokio::test]
    async fn test_server_handle_prompts_get_not_found() {
        let server = McpServer::new("test", "1.0");
        let msg = JsonRpcMessage::request(5, "prompts/get", serde_json::json!({"name": "unknown"}));
        let resp = server.handle_message(msg).await.unwrap();
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32602);
    }

    #[tokio::test]
    async fn test_server_handle_ping() {
        let server = McpServer::new("test", "1.0");
        let msg = JsonRpcMessage::request(6, "ping", serde_json::json!({}));
        let resp = server.handle_message(msg).await.unwrap();
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
    }

    #[tokio::test]
    async fn test_server_handle_unknown_method() {
        let server = McpServer::new("test", "1.0");
        let msg = JsonRpcMessage::request(7, "unknown/method", serde_json::json!({}));
        let resp = server.handle_message(msg).await.unwrap();
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32601);
    }

    #[tokio::test]
    async fn test_server_handle_resources_list() {
        let mut server = McpServer::new("test", "1.0");
        server.add_resource(McpResource {
            uri: "test://file".into(),
            name: "file".into(),
            description: None,
            mime_type: None,
        });
        let msg = JsonRpcMessage::request(8, "resources/list", serde_json::json!({}));
        let resp = server.handle_message(msg).await.unwrap();
        let resources = resp.result.unwrap()["resources"].as_array().unwrap().len();
        assert_eq!(resources, 1);
    }
}
