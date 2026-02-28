use gauss_core::agent::{Agent as RustAgent, StopCondition};
use gauss_core::context;
use gauss_core::eval;
use gauss_core::hitl;
use gauss_core::mcp;
use gauss_core::memory;
use gauss_core::message::Message as RustMessage;
use gauss_core::network;
use gauss_core::provider::anthropic::AnthropicProvider;
use gauss_core::provider::deepseek::DeepSeekProvider;
use gauss_core::provider::google::GoogleProvider;
use gauss_core::provider::groq::GroqProvider;
use gauss_core::provider::ollama::OllamaProvider;
use gauss_core::provider::openai::OpenAiProvider;
use gauss_core::provider::retry::{RetryConfig, RetryProvider};
use gauss_core::provider::{GenerateOptions, Provider, ProviderConfig};
use gauss_core::rag;
use gauss_core::telemetry;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde_json::json;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

fn providers() -> &'static Mutex<HashMap<u32, Arc<dyn Provider>>> {
    static PROVIDERS: OnceLock<Mutex<HashMap<u32, Arc<dyn Provider>>>> = OnceLock::new();
    PROVIDERS.get_or_init(|| Mutex::new(HashMap::new()))
}

static NEXT_ID: AtomicU32 = AtomicU32::new(1);

/// Gauss Core version.
#[pyfunction]
fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Create a provider. Returns handle ID.
#[pyfunction]
#[pyo3(signature = (provider_type, model, api_key, base_url=None, max_retries=None))]
fn create_provider(
    provider_type: &str,
    model: &str,
    api_key: &str,
    base_url: Option<String>,
    max_retries: Option<u32>,
) -> PyResult<u32> {
    let mut config = ProviderConfig::new(api_key);
    if let Some(url) = base_url {
        config.base_url = Some(url);
    }
    config.max_retries = max_retries;

    let retries = config.max_retries.unwrap_or(3);

    let inner: Arc<dyn Provider> = match provider_type {
        "openai" => Arc::new(OpenAiProvider::new(model, config)),
        "anthropic" => Arc::new(AnthropicProvider::new(model, config)),
        "google" => Arc::new(GoogleProvider::new(model, config)),
        "groq" => Arc::new(GroqProvider::create(model, config)),
        "ollama" => Arc::new(OllamaProvider::create(model, config)),
        "deepseek" => Arc::new(DeepSeekProvider::create(model, config)),
        other => {
            return Err(PyRuntimeError::new_err(format!(
                "Unknown provider: {other}"
            )));
        }
    };

    let provider: Arc<dyn Provider> = if retries > 0 {
        Arc::new(RetryProvider::new(
            inner,
            RetryConfig {
                max_retries: retries,
                ..RetryConfig::default()
            },
        ))
    } else {
        inner
    };

    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    providers().lock().unwrap().insert(id, provider);
    Ok(id)
}

/// Destroy a provider.
#[pyfunction]
fn destroy_provider(handle: u32) -> PyResult<()> {
    providers()
        .lock()
        .unwrap()
        .remove(&handle)
        .ok_or_else(|| PyRuntimeError::new_err(format!("Provider {handle} not found")))?;
    Ok(())
}

fn get_provider(handle: u32) -> PyResult<Arc<dyn Provider>> {
    providers()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| PyRuntimeError::new_err(format!("Provider {handle} not found")))
}

fn parse_messages(messages_json: &str) -> PyResult<Vec<RustMessage>> {
    let js_msgs: Vec<serde_json::Value> = serde_json::from_str(messages_json)
        .map_err(|e| PyRuntimeError::new_err(format!("Invalid messages JSON: {e}")))?;
    Ok(js_msgs
        .iter()
        .map(|m| {
            let role = m["role"].as_str().unwrap_or("user");
            let content = m["content"].as_str().unwrap_or("");
            match role {
                "system" => RustMessage::system(content),
                "assistant" => RustMessage::assistant(content),
                _ => RustMessage::user(content),
            }
        })
        .collect())
}

/// Call generate. Returns JSON string.
#[pyfunction]
#[pyo3(signature = (provider_handle, messages_json, temperature=None, max_tokens=None))]
fn generate(
    py: Python<'_>,
    provider_handle: u32,
    messages_json: String,
    temperature: Option<f64>,
    max_tokens: Option<u32>,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let provider = get_provider(provider_handle)?;
        let rust_msgs = parse_messages(&messages_json)?;

        let opts = GenerateOptions {
            temperature,
            max_tokens,
            ..GenerateOptions::default()
        };

        let result = provider
            .generate(&rust_msgs, &[], &opts)
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("Generate error: {e}")))?;

        let text = result.text().unwrap_or("").to_string();

        let output = json!({
            "text": text,
            "usage": {
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
            },
            "finish_reason": format!("{:?}", result.finish_reason),
        });

        serde_json::to_string(&output)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {e}")))
    })
}

/// Run an agent. Returns JSON string.
#[pyfunction]
#[pyo3(signature = (name, provider_handle, messages_json, options_json=None))]
fn agent_run(
    py: Python<'_>,
    name: String,
    provider_handle: u32,
    messages_json: String,
    options_json: Option<String>,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let provider = get_provider(provider_handle)?;
        let rust_msgs = parse_messages(&messages_json)?;
        let mut builder = RustAgent::builder(name, provider);

        if let Some(opts_str) = options_json {
            let opts: serde_json::Value = serde_json::from_str(&opts_str)
                .map_err(|e| PyRuntimeError::new_err(format!("Invalid options JSON: {e}")))?;

            if let Some(instructions) = opts["instructions"].as_str() {
                builder = builder.instructions(instructions);
            }
            if let Some(max_steps) = opts["max_steps"].as_u64() {
                builder = builder.max_steps(max_steps as usize);
            }
            if let Some(temp) = opts["temperature"].as_f64() {
                builder = builder.temperature(temp);
            }
            if let Some(tp) = opts["top_p"].as_f64() {
                builder = builder.top_p(tp);
            }
            if let Some(mt) = opts["max_tokens"].as_u64() {
                builder = builder.max_tokens(mt as u32);
            }
            if let Some(stop_tool) = opts["stop_on_tool"].as_str() {
                builder = builder.stop_when(StopCondition::HasToolCall(stop_tool.to_string()));
            }
            if let Some(schema) = opts.get("output_schema").filter(|s| !s.is_null()) {
                builder = builder.output_schema(schema.clone());
            }
        }

        let agent = builder.build();
        let output = agent
            .run(rust_msgs)
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("Agent error: {e}")))?;

        let result = json!({
            "text": output.text,
            "steps": output.steps,
            "usage": {
                "input_tokens": output.usage.input_tokens,
                "output_tokens": output.usage.output_tokens,
            },
            "structured_output": output.structured_output,
        });

        serde_json::to_string(&result)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {e}")))
    })
}

fn py_err(msg: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(format!("{msg}"))
}

// ============ Registries ============

fn memories() -> &'static Mutex<HashMap<u32, Arc<memory::InMemoryMemory>>> {
    static R: OnceLock<Mutex<HashMap<u32, Arc<memory::InMemoryMemory>>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

fn vector_stores() -> &'static Mutex<HashMap<u32, Arc<rag::InMemoryVectorStore>>> {
    static R: OnceLock<Mutex<HashMap<u32, Arc<rag::InMemoryVectorStore>>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

fn mcp_servers() -> &'static Mutex<HashMap<u32, Arc<tokio::sync::Mutex<mcp::McpServer>>>> {
    static R: OnceLock<Mutex<HashMap<u32, Arc<tokio::sync::Mutex<mcp::McpServer>>>>> =
        OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

fn networks() -> &'static Mutex<HashMap<u32, Arc<tokio::sync::Mutex<network::AgentNetwork>>>> {
    static R: OnceLock<Mutex<HashMap<u32, Arc<tokio::sync::Mutex<network::AgentNetwork>>>>> =
        OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

fn approvals() -> &'static Mutex<HashMap<u32, Arc<Mutex<hitl::ApprovalManager>>>> {
    static R: OnceLock<Mutex<HashMap<u32, Arc<Mutex<hitl::ApprovalManager>>>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

fn checkpoints() -> &'static Mutex<HashMap<u32, Arc<hitl::InMemoryCheckpointStore>>> {
    static R: OnceLock<Mutex<HashMap<u32, Arc<hitl::InMemoryCheckpointStore>>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

fn evals() -> &'static Mutex<HashMap<u32, Arc<Mutex<eval::EvalRunner>>>> {
    static R: OnceLock<Mutex<HashMap<u32, Arc<Mutex<eval::EvalRunner>>>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

fn collectors() -> &'static Mutex<HashMap<u32, Arc<Mutex<telemetry::TelemetryCollector>>>> {
    static R: OnceLock<Mutex<HashMap<u32, Arc<Mutex<telemetry::TelemetryCollector>>>>> =
        OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

// ============ Memory ============

#[pyfunction]
fn create_memory() -> u32 {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    memories()
        .lock()
        .unwrap()
        .insert(id, Arc::new(memory::InMemoryMemory::new()));
    id
}

#[pyfunction]
fn memory_store(
    py: Python<'_>,
    handle: u32,
    entry_json: String,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        use memory::Memory;
        let mem = memories()
            .lock()
            .unwrap()
            .get(&handle)
            .cloned()
            .ok_or_else(|| py_err("Memory not found"))?;
        let entry: memory::MemoryEntry =
            serde_json::from_str(&entry_json).map_err(|e| py_err(e))?;
        mem.store(entry).await.map_err(|e| py_err(e))
    })
}

#[pyfunction]
#[pyo3(signature = (handle, options_json=None))]
fn memory_recall(
    py: Python<'_>,
    handle: u32,
    options_json: Option<String>,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        use memory::Memory;
        let mem = memories()
            .lock()
            .unwrap()
            .get(&handle)
            .cloned()
            .ok_or_else(|| py_err("Memory not found"))?;
        let opts: memory::RecallOptions = match options_json {
            Some(j) => serde_json::from_str(&j).map_err(|e| py_err(e))?,
            None => memory::RecallOptions::default(),
        };
        let entries = mem.recall(opts).await.map_err(|e| py_err(e))?;
        serde_json::to_string(&entries).map_err(|e| py_err(e))
    })
}

#[pyfunction]
#[pyo3(signature = (handle, session_id=None))]
fn memory_clear(
    py: Python<'_>,
    handle: u32,
    session_id: Option<String>,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let mem = memories()
            .lock()
            .unwrap()
            .get(&handle)
            .cloned()
            .ok_or_else(|| py_err("Memory not found"))?;
        memory::Memory::clear(&*mem, session_id.as_deref())
            .await
            .map_err(|e| py_err(e))
    })
}

#[pyfunction]
fn destroy_memory(handle: u32) -> PyResult<()> {
    memories()
        .lock()
        .unwrap()
        .remove(&handle)
        .ok_or_else(|| py_err("Memory not found"))?;
    Ok(())
}

// ============ Context ============

#[pyfunction]
fn count_tokens(text: &str) -> u32 {
    context::count_tokens(text) as u32
}

#[pyfunction]
fn count_tokens_for_model(text: &str, model: &str) -> u32 {
    context::count_tokens_for_model(text, model) as u32
}

#[pyfunction]
fn count_message_tokens(messages_json: &str) -> PyResult<u32> {
    let msgs = parse_messages(messages_json)?;
    Ok(context::count_messages_tokens(&msgs) as u32)
}

#[pyfunction]
fn get_context_window_size(model: &str) -> u32 {
    context::context_window_size(model) as u32
}

// ============ RAG ============

#[pyfunction]
fn create_vector_store() -> u32 {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    vector_stores()
        .lock()
        .unwrap()
        .insert(id, Arc::new(rag::InMemoryVectorStore::new()));
    id
}

#[pyfunction]
fn vector_store_upsert(
    py: Python<'_>,
    handle: u32,
    chunks_json: String,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        use rag::VectorStore;
        let store = vector_stores()
            .lock()
            .unwrap()
            .get(&handle)
            .cloned()
            .ok_or_else(|| py_err("VectorStore not found"))?;
        let chunks: Vec<rag::Chunk> = serde_json::from_str(&chunks_json).map_err(|e| py_err(e))?;
        store.upsert(chunks).await.map_err(|e| py_err(e))
    })
}

#[pyfunction]
fn vector_store_search(
    py: Python<'_>,
    handle: u32,
    embedding_json: String,
    top_k: u32,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        use rag::VectorStore;
        let store = vector_stores()
            .lock()
            .unwrap()
            .get(&handle)
            .cloned()
            .ok_or_else(|| py_err("VectorStore not found"))?;
        let embedding: Vec<f32> = serde_json::from_str(&embedding_json).map_err(|e| py_err(e))?;
        let results = store
            .search(&embedding, top_k as usize)
            .await
            .map_err(|e| py_err(e))?;
        serde_json::to_string(&results).map_err(|e| py_err(e))
    })
}

#[pyfunction]
fn destroy_vector_store(handle: u32) -> PyResult<()> {
    vector_stores()
        .lock()
        .unwrap()
        .remove(&handle)
        .ok_or_else(|| py_err("VectorStore not found"))?;
    Ok(())
}

#[pyfunction]
fn cosine_similarity(a_json: &str, b_json: &str) -> PyResult<f64> {
    let a: Vec<f32> = serde_json::from_str(a_json).map_err(|e| py_err(e))?;
    let b: Vec<f32> = serde_json::from_str(b_json).map_err(|e| py_err(e))?;
    Ok(rag::cosine_similarity(&a, &b) as f64)
}

// ============ MCP ============

#[pyfunction]
fn create_mcp_server(name: &str, version_str: &str) -> u32 {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    mcp_servers().lock().unwrap().insert(
        id,
        Arc::new(tokio::sync::Mutex::new(mcp::McpServer::new(
            name,
            version_str,
        ))),
    );
    id
}

#[pyfunction]
fn mcp_server_add_tool(handle: u32, tool_json: String) -> PyResult<()> {
    let server = mcp_servers()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| py_err("McpServer not found"))?;
    let mcp_tool: mcp::McpTool = serde_json::from_str(&tool_json).map_err(|e| py_err(e))?;
    let gauss_tool = mcp::mcp_tool_to_gauss(&mcp_tool);
    server.blocking_lock().add_tool(gauss_tool);
    Ok(())
}

#[pyfunction]
fn mcp_server_handle(
    py: Python<'_>,
    handle: u32,
    message_json: String,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let server = mcp_servers()
            .lock()
            .unwrap()
            .get(&handle)
            .cloned()
            .ok_or_else(|| py_err("McpServer not found"))?;
        let msg: mcp::JsonRpcMessage =
            serde_json::from_str(&message_json).map_err(|e| py_err(e))?;
        let resp = server
            .lock()
            .await
            .handle_message(msg)
            .await
            .map_err(|e| py_err(e))?;
        serde_json::to_string(&resp).map_err(|e| py_err(e))
    })
}

#[pyfunction]
fn destroy_mcp_server(handle: u32) -> PyResult<()> {
    mcp_servers()
        .lock()
        .unwrap()
        .remove(&handle)
        .ok_or_else(|| py_err("McpServer not found"))?;
    Ok(())
}

// ============ Network ============

#[pyfunction]
fn create_network() -> u32 {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    networks().lock().unwrap().insert(
        id,
        Arc::new(tokio::sync::Mutex::new(network::AgentNetwork::new())),
    );
    id
}

#[pyfunction]
#[pyo3(signature = (handle, name, provider_handle, card_json=None, connections=None))]
fn network_add_agent(
    handle: u32,
    name: String,
    provider_handle: u32,
    card_json: Option<String>,
    connections: Option<Vec<String>>,
) -> PyResult<()> {
    let net = networks()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| py_err("Network not found"))?;
    let provider = get_provider(provider_handle)?;
    let card: network::AgentCard = match card_json {
        Some(j) => serde_json::from_str(&j).map_err(|e| py_err(e))?,
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
fn network_set_supervisor(handle: u32, name: String) -> PyResult<()> {
    let net = networks()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| py_err("Network not found"))?;
    net.blocking_lock().set_supervisor(name);
    Ok(())
}

#[pyfunction]
fn network_delegate(
    py: Python<'_>,
    handle: u32,
    agent_name: String,
    messages_json: String,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let net = networks()
            .lock()
            .unwrap()
            .get(&handle)
            .cloned()
            .ok_or_else(|| py_err("Network not found"))?;
        let msgs = parse_messages(&messages_json)?;
        let output = net
            .lock()
            .await
            .delegate(&agent_name, msgs)
            .await
            .map_err(|e| py_err(e))?;
        serde_json::to_string(&output).map_err(|e| py_err(e))
    })
}

#[pyfunction]
fn destroy_network(handle: u32) -> PyResult<()> {
    networks()
        .lock()
        .unwrap()
        .remove(&handle)
        .ok_or_else(|| py_err("Network not found"))?;
    Ok(())
}

// ============ HITL ============

#[pyfunction]
fn create_approval_manager() -> u32 {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    approvals()
        .lock()
        .unwrap()
        .insert(id, Arc::new(Mutex::new(hitl::ApprovalManager::new())));
    id
}

#[pyfunction]
fn approval_request(
    handle: u32,
    tool_name: String,
    args_json: String,
    session_id: String,
) -> PyResult<String> {
    let mgr = approvals()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| py_err("ApprovalManager not found"))?;
    let args: serde_json::Value = serde_json::from_str(&args_json).map_err(|e| py_err(e))?;
    let req = mgr
        .lock()
        .unwrap()
        .request_approval(tool_name, args, 0, session_id)
        .map_err(|e| py_err(e))?;
    Ok(req.id.clone())
}

#[pyfunction]
#[pyo3(signature = (handle, request_id, modified_args=None))]
fn approval_approve(handle: u32, request_id: &str, modified_args: Option<String>) -> PyResult<()> {
    let mgr = approvals()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| py_err("ApprovalManager not found"))?;
    let args: Option<serde_json::Value> = match modified_args {
        Some(j) => Some(serde_json::from_str(&j).map_err(|e| py_err(e))?),
        None => None,
    };
    mgr.lock()
        .unwrap()
        .approve(request_id, args)
        .map_err(|e| py_err(e))?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (handle, request_id, reason=None))]
fn approval_deny(handle: u32, request_id: &str, reason: Option<String>) -> PyResult<()> {
    let mgr = approvals()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| py_err("ApprovalManager not found"))?;
    mgr.lock()
        .unwrap()
        .deny(request_id, reason)
        .map_err(|e| py_err(e))?;
    Ok(())
}

#[pyfunction]
fn approval_list_pending(handle: u32) -> PyResult<String> {
    let mgr = approvals()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| py_err("ApprovalManager not found"))?;
    let pending = mgr.lock().unwrap().list_pending().map_err(|e| py_err(e))?;
    serde_json::to_string(&pending).map_err(|e| py_err(e))
}

#[pyfunction]
fn destroy_approval_manager(handle: u32) -> PyResult<()> {
    approvals()
        .lock()
        .unwrap()
        .remove(&handle)
        .ok_or_else(|| py_err("ApprovalManager not found"))?;
    Ok(())
}

// ============ Checkpoint Store ============

#[pyfunction]
fn create_checkpoint_store() -> u32 {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    checkpoints()
        .lock()
        .unwrap()
        .insert(id, Arc::new(hitl::InMemoryCheckpointStore::new()));
    id
}

#[pyfunction]
fn checkpoint_save(
    py: Python<'_>,
    handle: u32,
    checkpoint_json: String,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        use hitl::CheckpointStore;
        let store = checkpoints()
            .lock()
            .unwrap()
            .get(&handle)
            .cloned()
            .ok_or_else(|| py_err("CheckpointStore not found"))?;
        let cp: hitl::Checkpoint = serde_json::from_str(&checkpoint_json).map_err(|e| py_err(e))?;
        store.save(&cp).await.map_err(|e| py_err(e))
    })
}

#[pyfunction]
fn checkpoint_load(
    py: Python<'_>,
    handle: u32,
    checkpoint_id: String,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        use hitl::CheckpointStore;
        let store = checkpoints()
            .lock()
            .unwrap()
            .get(&handle)
            .cloned()
            .ok_or_else(|| py_err("CheckpointStore not found"))?;
        let cp = store.load(&checkpoint_id).await.map_err(|e| py_err(e))?;
        serde_json::to_string(&cp).map_err(|e| py_err(e))
    })
}

#[pyfunction]
fn destroy_checkpoint_store(handle: u32) -> PyResult<()> {
    checkpoints()
        .lock()
        .unwrap()
        .remove(&handle)
        .ok_or_else(|| py_err("CheckpointStore not found"))?;
    Ok(())
}

// ============ Eval ============

#[pyfunction]
#[pyo3(signature = (threshold=None))]
fn create_eval_runner(threshold: Option<f64>) -> u32 {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let mut runner = eval::EvalRunner::new();
    if let Some(t) = threshold {
        runner = runner.with_threshold(t);
    }
    evals()
        .lock()
        .unwrap()
        .insert(id, Arc::new(Mutex::new(runner)));
    id
}

#[pyfunction]
fn eval_add_scorer(handle: u32, scorer_type: &str) -> PyResult<()> {
    let runner = evals()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| py_err("EvalRunner not found"))?;
    let scorer: Arc<dyn eval::Scorer> = match scorer_type {
        "exact_match" => Arc::new(eval::ExactMatchScorer),
        "contains" => Arc::new(eval::ContainsScorer),
        "length_ratio" => Arc::new(eval::LengthRatioScorer),
        other => return Err(py_err(&format!("Unknown scorer: {other}"))),
    };
    runner.lock().unwrap().add_scorer(scorer);
    Ok(())
}

#[pyfunction]
fn load_dataset_jsonl(jsonl: &str) -> PyResult<String> {
    let cases = eval::load_dataset_jsonl(jsonl).map_err(|e| py_err(e))?;
    serde_json::to_string(&cases).map_err(|e| py_err(e))
}

#[pyfunction]
fn load_dataset_json(json_str: &str) -> PyResult<String> {
    let cases = eval::load_dataset_json(json_str).map_err(|e| py_err(e))?;
    serde_json::to_string(&cases).map_err(|e| py_err(e))
}

#[pyfunction]
fn destroy_eval_runner(handle: u32) -> PyResult<()> {
    evals()
        .lock()
        .unwrap()
        .remove(&handle)
        .ok_or_else(|| py_err("EvalRunner not found"))?;
    Ok(())
}

// ============ Telemetry ============

#[pyfunction]
fn create_telemetry() -> u32 {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    collectors().lock().unwrap().insert(
        id,
        Arc::new(Mutex::new(telemetry::TelemetryCollector::new())),
    );
    id
}

#[pyfunction]
fn telemetry_record_span(handle: u32, span_json: &str) -> PyResult<()> {
    let coll = collectors()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| py_err("TelemetryCollector not found"))?;
    let span: telemetry::SpanRecord = serde_json::from_str(span_json).map_err(|e| py_err(e))?;
    coll.lock().unwrap().record_span(span);
    Ok(())
}

#[pyfunction]
fn telemetry_export_spans(handle: u32) -> PyResult<String> {
    let coll = collectors()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| py_err("TelemetryCollector not found"))?;
    let spans = coll.lock().unwrap().export_spans();
    serde_json::to_string(&spans).map_err(|e| py_err(e))
}

#[pyfunction]
fn telemetry_export_metrics(handle: u32) -> PyResult<String> {
    let coll = collectors()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| py_err("TelemetryCollector not found"))?;
    let metrics = coll.lock().unwrap().export_metrics();
    serde_json::to_string(&metrics).map_err(|e| py_err(e))
}

#[pyfunction]
fn telemetry_clear(handle: u32) -> PyResult<()> {
    let coll = collectors()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| py_err("TelemetryCollector not found"))?;
    coll.lock().unwrap().clear();
    Ok(())
}

#[pyfunction]
fn destroy_telemetry(handle: u32) -> PyResult<()> {
    collectors()
        .lock()
        .unwrap()
        .remove(&handle)
        .ok_or_else(|| py_err("TelemetryCollector not found"))?;
    Ok(())
}

/// Gauss Core Python module.
#[pymodule]
#[pyo3(name = "gauss_core")]
fn gauss_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(create_provider, m)?)?;
    m.add_function(wrap_pyfunction!(destroy_provider, m)?)?;
    m.add_function(wrap_pyfunction!(generate, m)?)?;
    m.add_function(wrap_pyfunction!(agent_run, m)?)?;
    // Memory
    m.add_function(wrap_pyfunction!(create_memory, m)?)?;
    m.add_function(wrap_pyfunction!(memory_store, m)?)?;
    m.add_function(wrap_pyfunction!(memory_recall, m)?)?;
    m.add_function(wrap_pyfunction!(memory_clear, m)?)?;
    m.add_function(wrap_pyfunction!(destroy_memory, m)?)?;
    // Context
    m.add_function(wrap_pyfunction!(count_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(count_tokens_for_model, m)?)?;
    m.add_function(wrap_pyfunction!(count_message_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(get_context_window_size, m)?)?;
    // RAG
    m.add_function(wrap_pyfunction!(create_vector_store, m)?)?;
    m.add_function(wrap_pyfunction!(vector_store_upsert, m)?)?;
    m.add_function(wrap_pyfunction!(vector_store_search, m)?)?;
    m.add_function(wrap_pyfunction!(destroy_vector_store, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    // MCP
    m.add_function(wrap_pyfunction!(create_mcp_server, m)?)?;
    m.add_function(wrap_pyfunction!(mcp_server_add_tool, m)?)?;
    m.add_function(wrap_pyfunction!(mcp_server_handle, m)?)?;
    m.add_function(wrap_pyfunction!(destroy_mcp_server, m)?)?;
    // Network
    m.add_function(wrap_pyfunction!(create_network, m)?)?;
    m.add_function(wrap_pyfunction!(network_add_agent, m)?)?;
    m.add_function(wrap_pyfunction!(network_set_supervisor, m)?)?;
    m.add_function(wrap_pyfunction!(network_delegate, m)?)?;
    m.add_function(wrap_pyfunction!(destroy_network, m)?)?;
    // HITL
    m.add_function(wrap_pyfunction!(create_approval_manager, m)?)?;
    m.add_function(wrap_pyfunction!(approval_request, m)?)?;
    m.add_function(wrap_pyfunction!(approval_approve, m)?)?;
    m.add_function(wrap_pyfunction!(approval_deny, m)?)?;
    m.add_function(wrap_pyfunction!(approval_list_pending, m)?)?;
    m.add_function(wrap_pyfunction!(destroy_approval_manager, m)?)?;
    // Checkpoint
    m.add_function(wrap_pyfunction!(create_checkpoint_store, m)?)?;
    m.add_function(wrap_pyfunction!(checkpoint_save, m)?)?;
    m.add_function(wrap_pyfunction!(checkpoint_load, m)?)?;
    m.add_function(wrap_pyfunction!(destroy_checkpoint_store, m)?)?;
    // Eval
    m.add_function(wrap_pyfunction!(create_eval_runner, m)?)?;
    m.add_function(wrap_pyfunction!(eval_add_scorer, m)?)?;
    m.add_function(wrap_pyfunction!(load_dataset_jsonl, m)?)?;
    m.add_function(wrap_pyfunction!(load_dataset_json, m)?)?;
    m.add_function(wrap_pyfunction!(destroy_eval_runner, m)?)?;
    // Telemetry
    m.add_function(wrap_pyfunction!(create_telemetry, m)?)?;
    m.add_function(wrap_pyfunction!(telemetry_record_span, m)?)?;
    m.add_function(wrap_pyfunction!(telemetry_export_spans, m)?)?;
    m.add_function(wrap_pyfunction!(telemetry_export_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(telemetry_clear, m)?)?;
    m.add_function(wrap_pyfunction!(destroy_telemetry, m)?)?;
    Ok(())
}
