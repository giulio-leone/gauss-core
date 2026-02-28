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
