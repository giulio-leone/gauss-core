use crate::provider::get_provider;
use crate::registry::HandleRegistry;
use gauss_core::agent::Agent as RustAgent;
use gauss_core::message::Message as RustMessage;
use gauss_core::network;
use napi::bindgen_prelude::*;
use serde_json::json;
use std::sync::Arc;

// ============ Network ============

static NETWORKS: HandleRegistry<Arc<tokio::sync::Mutex<network::AgentNetwork>>> =
    HandleRegistry::new();

#[napi]
pub fn create_network() -> u32 {
    NETWORKS.insert(Arc::new(tokio::sync::Mutex::new(
        network::AgentNetwork::new(),
    )))
}

#[napi]
pub fn network_add_agent(
    handle: u32,
    name: String,
    provider_handle: u32,
    instructions: Option<String>,
) -> Result<()> {
    let provider = get_provider(provider_handle)?;
    let mut builder = RustAgent::builder(&name, provider);
    if let Some(instr) = instructions {
        builder = builder.instructions(instr);
    }
    let agent = builder.build();
    let node = network::AgentNode {
        agent,
        card: network::AgentCard {
            name: name.clone(),
            ..Default::default()
        },
        connections: Vec::new(),
    };

    let net = NETWORKS.get_clone(handle)?;
    net.blocking_lock().add_agent(node);
    Ok(())
}

#[napi]
pub fn network_set_supervisor(handle: u32, agent_name: String) -> Result<()> {
    let net = NETWORKS.get_clone(handle)?;
    net.blocking_lock().set_supervisor(agent_name);
    Ok(())
}

#[napi]
pub async fn network_delegate(
    handle: u32,
    agent_name: String,
    prompt: String,
) -> Result<serde_json::Value> {
    let net = NETWORKS.get_clone(handle)?;
    let messages = vec![RustMessage::user(&prompt)];
    let result = net
        .lock()
        .await
        .delegate(&agent_name, messages)
        .await
        .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
    Ok(json!({
        "agentName": result.agent_name,
        "resultText": result.result_text,
        "success": result.success,
        "error": result.error,
    }))
}

#[napi]
pub fn network_agent_cards(handle: u32) -> Result<serde_json::Value> {
    let net = NETWORKS.get_clone(handle)?;
    let net_guard = net.blocking_lock();
    let cards = net_guard.agent_cards();
    serde_json::to_value(&cards).map_err(|e| napi::Error::from_reason(format!("{e}")))
}

#[napi]
pub fn destroy_network(handle: u32) -> Result<()> {
    NETWORKS.remove(handle)?;
    Ok(())
}

// ============ A2A Protocol ============

#[napi]
pub fn create_a2a_client(base_url: String, auth_token: Option<String>) -> Result<serde_json::Value> {
    let mut config = json!({ "baseUrl": base_url });
    if let Some(token) = &auth_token {
        config["authToken"] = json!(token);
    }
    Ok(config)
}

#[napi]
pub async fn a2a_discover(base_url: String, auth_token: Option<String>) -> Result<serde_json::Value> {
    let mut client = gauss_core::a2a_client::A2aClient::new(&base_url);
    if let Some(token) = auth_token {
        client = client.with_auth_token(&token);
    }
    let card = client
        .discover()
        .await
        .map_err(|e| napi::Error::from_reason(format!("A2A discover error: {e}")))?;
    serde_json::to_value(&card).map_err(|e| napi::Error::from_reason(format!("Serialize error: {e}")))
}

#[napi]
pub async fn a2a_send_message(
    base_url: String,
    auth_token: Option<String>,
    message_json: String,
    config_json: Option<String>,
) -> Result<serde_json::Value> {
    let mut client = gauss_core::a2a_client::A2aClient::new(&base_url);
    if let Some(token) = auth_token {
        client = client.with_auth_token(&token);
    }
    let message: gauss_core::a2a::A2aMessage = serde_json::from_str(&message_json)
        .map_err(|e| napi::Error::from_reason(format!("Invalid message JSON: {e}")))?;
    let config = config_json
        .map(|c| serde_json::from_str(&c))
        .transpose()
        .map_err(|e| napi::Error::from_reason(format!("Invalid config JSON: {e}")))?;
    let result = client
        .send_message(message, config)
        .await
        .map_err(|e| napi::Error::from_reason(format!("A2A send error: {e}")))?;
    let value = match result {
        gauss_core::a2a_client::SendMessageResult::Task(task) => {
            let mut v = serde_json::to_value(&task)
                .map_err(|e| napi::Error::from_reason(format!("Serialize error: {e}")))?;
            v["_type"] = json!("task");
            v
        }
        gauss_core::a2a_client::SendMessageResult::Message(msg) => {
            let mut v = serde_json::to_value(&msg)
                .map_err(|e| napi::Error::from_reason(format!("Serialize error: {e}")))?;
            v["_type"] = json!("message");
            v
        }
    };
    Ok(value)
}

#[napi]
pub async fn a2a_ask(
    base_url: String,
    auth_token: Option<String>,
    text: String,
) -> Result<String> {
    let mut client = gauss_core::a2a_client::A2aClient::new(&base_url);
    if let Some(token) = auth_token {
        client = client.with_auth_token(&token);
    }
    client
        .ask(&text)
        .await
        .map_err(|e| napi::Error::from_reason(format!("A2A ask error: {e}")))
}

#[napi]
pub async fn a2a_get_task(
    base_url: String,
    auth_token: Option<String>,
    task_id: String,
    history_length: Option<u32>,
) -> Result<serde_json::Value> {
    let mut client = gauss_core::a2a_client::A2aClient::new(&base_url);
    if let Some(token) = auth_token {
        client = client.with_auth_token(&token);
    }
    let task = client
        .get_task(&task_id, history_length)
        .await
        .map_err(|e| napi::Error::from_reason(format!("A2A get_task error: {e}")))?;
    serde_json::to_value(&task).map_err(|e| napi::Error::from_reason(format!("Serialize error: {e}")))
}

#[napi]
pub async fn a2a_cancel_task(
    base_url: String,
    auth_token: Option<String>,
    task_id: String,
) -> Result<serde_json::Value> {
    let mut client = gauss_core::a2a_client::A2aClient::new(&base_url);
    if let Some(token) = auth_token {
        client = client.with_auth_token(&token);
    }
    let task = client
        .cancel_task(&task_id)
        .await
        .map_err(|e| napi::Error::from_reason(format!("A2A cancel error: {e}")))?;
    serde_json::to_value(&task).map_err(|e| napi::Error::from_reason(format!("Serialize error: {e}")))
}

#[napi]
pub async fn a2a_handle_request(
    agent_card_json: String,
    request_body: String,
) -> Result<String> {
    let _card: gauss_core::a2a::AgentCard = serde_json::from_str(&agent_card_json)
        .map_err(|e| napi::Error::from_reason(format!("Invalid AgentCard JSON: {e}")))?;
    let _req: gauss_core::a2a::JsonRpcRequest = serde_json::from_str(&request_body)
        .map_err(|e| napi::Error::from_reason(format!("Invalid JSON-RPC request: {e}")))?;
    let resp = gauss_core::a2a::JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        id: _req.id,
        result: None,
        error: Some(gauss_core::a2a::JsonRpcError {
            code: -32601,
            message: "Use the JavaScript A2aServer class to handle requests".to_string(),
            data: None,
        }),
    };
    serde_json::to_string(&resp).map_err(|e| napi::Error::from_reason(format!("Serialize error: {e}")))
}
