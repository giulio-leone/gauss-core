use crate::provider::get_provider;
use crate::registry::{py_err, HandleRegistry};
use crate::types::parse_messages;
use gauss_core::agent::Agent as RustAgent;
use gauss_core::network;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::Arc;

// ============ Network ============

static NETWORKS: HandleRegistry<Arc<tokio::sync::Mutex<network::AgentNetwork>>> =
    HandleRegistry::new();

#[pyfunction]
pub fn create_network() -> u32 {
    NETWORKS.insert(Arc::new(tokio::sync::Mutex::new(
        network::AgentNetwork::new(),
    )))
}

#[pyfunction]
#[pyo3(signature = (handle, name, provider_handle, card_json=None, connections=None))]
pub fn network_add_agent(
    handle: u32,
    name: String,
    provider_handle: u32,
    card_json: Option<String>,
    connections: Option<Vec<String>>,
) -> PyResult<()> {
    let net = NETWORKS.get_clone(handle)?;
    let provider = get_provider(provider_handle)?;
    let card: network::AgentCard = match card_json {
        Some(j) => serde_json::from_str(&j).map_err(py_err)?,
        None => network::AgentCard {
            name: name.clone(),
            ..Default::default()
        },
    };
    let agent = RustAgent::builder(&name, provider).build();
    let node = network::AgentNode {
        agent,
        card,
        connections: connections.unwrap_or_default(),
    };
    net.blocking_lock().add_agent(node);
    Ok(())
}

#[pyfunction]
pub fn network_set_supervisor(handle: u32, name: String) -> PyResult<()> {
    let net = NETWORKS.get_clone(handle)?;
    net.blocking_lock().set_supervisor(name);
    Ok(())
}

#[pyfunction]
pub fn network_delegate(
    py: Python<'_>,
    handle: u32,
    agent_name: String,
    messages_json: String,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let net = NETWORKS.get_clone(handle)?;
        let msgs = parse_messages(&messages_json)?;
        let output = net
            .lock()
            .await
            .delegate(&agent_name, msgs)
            .await
            .map_err(py_err)?;
        serde_json::to_string(&output).map_err(py_err)
    })
}

#[pyfunction]
pub fn destroy_network(handle: u32) -> PyResult<()> {
    NETWORKS.remove(handle)
}

#[pyfunction]
pub fn network_agent_cards(handle: u32) -> PyResult<String> {
    let net = NETWORKS.get_clone(handle)?;
    let guard = net.blocking_lock();
    let cards = guard.agent_cards();
    serde_json::to_string(&cards).map_err(py_err)
}

// ============ A2A Protocol ============

/// Discover a remote A2A agent's capabilities.
#[pyfunction]
#[pyo3(signature = (base_url, auth_token=None))]
pub fn a2a_discover(
    py: Python<'_>,
    base_url: String,
    auth_token: Option<String>,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let mut client = gauss_core::a2a_client::A2aClient::new(&base_url);
        if let Some(token) = auth_token {
            client = client.with_auth_token(&token);
        }
        let card = client
            .discover()
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("A2A discover error: {e}")))?;
        serde_json::to_string(&card)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {e}")))
    })
}

/// Send a message to a remote A2A agent.
#[pyfunction]
#[pyo3(signature = (base_url, auth_token=None, message_json="".to_string(), config_json=None))]
pub fn a2a_send_message(
    py: Python<'_>,
    base_url: String,
    auth_token: Option<String>,
    message_json: String,
    config_json: Option<String>,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let mut client = gauss_core::a2a_client::A2aClient::new(&base_url);
        if let Some(token) = auth_token {
            client = client.with_auth_token(&token);
        }
        let message: gauss_core::a2a::A2aMessage = serde_json::from_str(&message_json)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid message JSON: {e}")))?;
        let config = config_json
            .map(|c| serde_json::from_str(&c))
            .transpose()
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid config JSON: {e}")))?;
        let result = client
            .send_message(message, config)
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("A2A send error: {e}")))?;
        let value = match result {
            gauss_core::a2a_client::SendMessageResult::Task(task) => {
                serde_json::to_string(&task)
            }
            gauss_core::a2a_client::SendMessageResult::Message(msg) => {
                serde_json::to_string(&msg)
            }
        };
        value.map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {e}")))
    })
}

/// Quick A2A helper: send text and get text response.
#[pyfunction]
#[pyo3(signature = (base_url, auth_token=None, text="".to_string()))]
pub fn a2a_ask(
    py: Python<'_>,
    base_url: String,
    auth_token: Option<String>,
    text: String,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let mut client = gauss_core::a2a_client::A2aClient::new(&base_url);
        if let Some(token) = auth_token {
            client = client.with_auth_token(&token);
        }
        client
            .ask(&text)
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("A2A ask error: {e}")))
    })
}

/// Get an A2A task by ID.
#[pyfunction]
#[pyo3(signature = (base_url, auth_token=None, task_id="".to_string(), history_length=None))]
pub fn a2a_get_task(
    py: Python<'_>,
    base_url: String,
    auth_token: Option<String>,
    task_id: String,
    history_length: Option<u32>,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let mut client = gauss_core::a2a_client::A2aClient::new(&base_url);
        if let Some(token) = auth_token {
            client = client.with_auth_token(&token);
        }
        let task = client
            .get_task(&task_id, history_length)
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("A2A get_task error: {e}")))?;
        serde_json::to_string(&task)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {e}")))
    })
}

/// Cancel an A2A task.
#[pyfunction]
#[pyo3(signature = (base_url, auth_token=None, task_id="".to_string()))]
pub fn a2a_cancel_task(
    py: Python<'_>,
    base_url: String,
    auth_token: Option<String>,
    task_id: String,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let mut client = gauss_core::a2a_client::A2aClient::new(&base_url);
        if let Some(token) = auth_token {
            client = client.with_auth_token(&token);
        }
        let task = client
            .cancel_task(&task_id)
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("A2A cancel error: {e}")))?;
        serde_json::to_string(&task)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {e}")))
    })
}
