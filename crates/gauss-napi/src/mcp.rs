use crate::registry::HandleRegistry;
use gauss_core::mcp;
use napi::bindgen_prelude::*;
use std::sync::Arc;

static MCP_SERVERS: HandleRegistry<Arc<tokio::sync::Mutex<mcp::McpServer>>> =
    HandleRegistry::new();

#[napi]
pub fn create_mcp_server(name: String, version_str: String) -> u32 {
    MCP_SERVERS.insert(Arc::new(tokio::sync::Mutex::new(mcp::McpServer::new(
        name,
        version_str,
    ))))
}

#[napi]
pub fn mcp_server_add_tool(handle: u32, tool_json: String) -> Result<()> {
    let mcp_tool: mcp::McpTool = serde_json::from_str(&tool_json)
        .map_err(|e| napi::Error::from_reason(format!("Invalid tool: {e}")))?;
    let gauss_tool = mcp::mcp_tool_to_gauss(&mcp_tool);
    let server = MCP_SERVERS.get_clone(handle)?;
    server.blocking_lock().add_tool(gauss_tool);
    Ok(())
}

#[napi]
pub fn mcp_server_add_resource(handle: u32, resource_json: String) -> Result<()> {
    let resource: mcp::McpResource = serde_json::from_str(&resource_json)
        .map_err(|e| napi::Error::from_reason(format!("Invalid resource: {e}")))?;
    let server = MCP_SERVERS.get_clone(handle)?;
    server.blocking_lock().add_resource(resource);
    Ok(())
}

#[napi]
pub fn mcp_server_add_prompt(handle: u32, prompt_json: String) -> Result<()> {
    let prompt: mcp::McpPrompt = serde_json::from_str(&prompt_json)
        .map_err(|e| napi::Error::from_reason(format!("Invalid prompt: {e}")))?;
    let server = MCP_SERVERS.get_clone(handle)?;
    server.blocking_lock().add_prompt(prompt);
    Ok(())
}

#[napi]
pub async fn mcp_server_handle(handle: u32, message_json: String) -> Result<serde_json::Value> {
    let server = MCP_SERVERS.get_clone(handle)?;
    let msg: mcp::JsonRpcMessage = serde_json::from_str(&message_json)
        .map_err(|e| napi::Error::from_reason(format!("Invalid message: {e}")))?;
    let resp = server
        .lock()
        .await
        .handle_message(msg)
        .await
        .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
    serde_json::to_value(&resp).map_err(|e| napi::Error::from_reason(format!("{e}")))
}

#[napi]
pub fn destroy_mcp_server(handle: u32) -> Result<()> {
    MCP_SERVERS.remove(handle)?;
    Ok(())
}
