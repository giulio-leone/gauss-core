use crate::registry::{py_err, HandleRegistry};
use gauss_core::mcp;
use pyo3::prelude::*;
use std::sync::Arc;

static MCP_SERVERS: HandleRegistry<Arc<tokio::sync::Mutex<mcp::McpServer>>> =
    HandleRegistry::new();

#[pyfunction]
pub fn create_mcp_server(name: &str, version_str: &str) -> u32 {
    MCP_SERVERS.insert(Arc::new(tokio::sync::Mutex::new(mcp::McpServer::new(
        name,
        version_str,
    ))))
}

#[pyfunction]
pub fn mcp_server_add_tool(handle: u32, tool_json: String) -> PyResult<()> {
    let server = MCP_SERVERS.get_clone(handle)?;
    let mcp_tool: mcp::McpTool = serde_json::from_str(&tool_json).map_err(py_err)?;
    let gauss_tool = mcp::mcp_tool_to_gauss(&mcp_tool);
    server.blocking_lock().add_tool(gauss_tool);
    Ok(())
}

#[pyfunction]
pub fn mcp_server_add_resource(handle: u32, resource_json: String) -> PyResult<()> {
    let server = MCP_SERVERS.get_clone(handle)?;
    let resource: mcp::McpResource = serde_json::from_str(&resource_json).map_err(py_err)?;
    server.blocking_lock().add_resource(resource);
    Ok(())
}

#[pyfunction]
pub fn mcp_server_add_prompt(handle: u32, prompt_json: String) -> PyResult<()> {
    let server = MCP_SERVERS.get_clone(handle)?;
    let prompt: mcp::McpPrompt = serde_json::from_str(&prompt_json).map_err(py_err)?;
    server.blocking_lock().add_prompt(prompt);
    Ok(())
}

#[pyfunction]
pub fn mcp_server_handle(
    py: Python<'_>,
    handle: u32,
    message_json: String,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let server = MCP_SERVERS.get_clone(handle)?;
        let msg: mcp::JsonRpcMessage = serde_json::from_str(&message_json).map_err(py_err)?;
        let resp = server
            .lock()
            .await
            .handle_message(msg)
            .await
            .map_err(py_err)?;
        serde_json::to_string(&resp).map_err(py_err)
    })
}

#[pyfunction]
pub fn destroy_mcp_server(handle: u32) -> PyResult<()> {
    MCP_SERVERS.remove(handle)
}
