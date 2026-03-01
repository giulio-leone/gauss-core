use gauss_core::agent::{Agent as RustAgent, StopCondition};
use gauss_core::context;
use gauss_core::eval;
use gauss_core::guardrail;
use gauss_core::hitl;
use gauss_core::mcp;
use gauss_core::memory::{self, Memory as _};
use gauss_core::message::Message as RustMessage;
use gauss_core::network;
use gauss_core::plugin;
use gauss_core::team;
use gauss_core::provider::anthropic::AnthropicProvider;
use gauss_core::provider::deepseek::DeepSeekProvider;
use gauss_core::provider::google::GoogleProvider;
use gauss_core::provider::groq::GroqProvider;
use gauss_core::provider::ollama::OllamaProvider;
use gauss_core::provider::openai::OpenAiProvider;
use gauss_core::provider::retry::{RetryConfig, RetryProvider};
use gauss_core::provider::{GenerateOptions, Provider, ProviderConfig};
use gauss_core::rag;
use gauss_core::resilience;
use gauss_core::stream_transform;
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
    providers()
        .lock()
        .expect("registry mutex poisoned")
        .insert(id, provider);
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

/// Get provider capabilities. Returns JSON string.
#[pyfunction]
fn get_provider_capabilities(provider_handle: u32) -> PyResult<String> {
    let provider = get_provider(provider_handle)?;
    let caps = provider.capabilities();
    let output = json!({
        "streaming": caps.streaming,
        "tool_use": caps.tool_use,
        "vision": caps.vision,
        "audio": caps.audio,
        "extended_thinking": caps.extended_thinking,
        "citations": caps.citations,
        "cache_control": caps.cache_control,
        "structured_output": caps.structured_output,
        "reasoning_effort": caps.reasoning_effort,
        "image_generation": caps.image_generation,
        "grounding": caps.grounding,
        "code_execution": caps.code_execution,
        "web_search": caps.web_search,
    });
    serde_json::to_string(&output)
        .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {e}")))
}

/// Call generate. Returns JSON string.
#[pyfunction]
#[pyo3(signature = (provider_handle, messages_json, temperature=None, max_tokens=None, thinking_budget=None, cache_control=None))]
fn generate(
    py: Python<'_>,
    provider_handle: u32,
    messages_json: String,
    temperature: Option<f64>,
    max_tokens: Option<u32>,
    thinking_budget: Option<u32>,
    cache_control: Option<bool>,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let provider = get_provider(provider_handle)?;
        let rust_msgs = parse_messages(&messages_json)?;

        let opts = GenerateOptions {
            temperature,
            max_tokens,
            thinking_budget,
            cache_control: cache_control.unwrap_or(false),
            ..GenerateOptions::default()
        };

        let result = provider
            .generate(&rust_msgs, &[], &opts)
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("Generate error: {e}")))?;

        let text = result.text().unwrap_or("").to_string();

        let citations_json: Vec<serde_json::Value> = result.citations.iter().map(|c| json!({
            "type": c.citation_type,
            "cited_text": c.cited_text,
            "document_title": c.document_title,
            "start": c.start,
            "end": c.end,
        })).collect();

        let output = json!({
            "text": text,
            "thinking": result.thinking,
            "citations": citations_json,
            "usage": {
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "cache_read_tokens": result.usage.cache_read_tokens,
                "cache_creation_tokens": result.usage.cache_creation_tokens,
            },
            "finish_reason": format!("{:?}", result.finish_reason),
        });

        serde_json::to_string(&output)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {e}")))
    })
}

/// Stream generate. Returns JSON array of StreamEvent objects.
#[pyfunction]
#[pyo3(signature = (provider_handle, messages_json, temperature=None, max_tokens=None))]
fn stream_generate(
    py: Python<'_>,
    provider_handle: u32,
    messages_json: String,
    temperature: Option<f64>,
    max_tokens: Option<u32>,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        use futures::StreamExt;
        let provider = get_provider(provider_handle)?;
        let rust_msgs = parse_messages(&messages_json)?;

        let opts = GenerateOptions {
            temperature,
            max_tokens,
            ..GenerateOptions::default()
        };

        let mut stream = provider
            .stream(&rust_msgs, &[], &opts)
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("Stream error: {e}")))?;

        let mut events = Vec::new();
        while let Some(event) = stream.next().await {
            match event {
                Ok(e) => {
                    let json = serde_json::to_string(&e)
                        .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
                    events.push(json);
                }
                Err(e) => {
                    return Err(PyRuntimeError::new_err(format!("Stream event error: {e}")));
                }
            }
        }

        let output = format!("[{}]", events.join(","));
        Ok(output)
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
            if let Some(budget) = opts["thinking_budget"].as_u64() {
                builder = builder.thinking_budget(budget as u32);
            }
            if opts["cache_control"].as_bool().unwrap_or(false) {
                builder = builder.cache_control(true);
            }

            // Code execution
            if let Some(ce) = opts.get("code_execution").filter(|v| !v.is_null()) {
                let ce_config = if ce.is_boolean() && ce.as_bool() == Some(true) {
                    gauss_core::code_execution::CodeExecutionConfig::all()
                } else if ce.is_object() {
                    let sandbox = match ce["sandbox"].as_str() {
                        Some("strict") => gauss_core::code_execution::SandboxConfig::strict(),
                        Some("permissive") => gauss_core::code_execution::SandboxConfig::permissive(),
                        _ => gauss_core::code_execution::SandboxConfig::default(),
                    };
                    gauss_core::code_execution::CodeExecutionConfig {
                        python: ce["python"].as_bool().unwrap_or(true),
                        javascript: ce["javascript"].as_bool().unwrap_or(true),
                        bash: ce["bash"].as_bool().unwrap_or(true),
                        timeout: std::time::Duration::from_secs(
                            ce["timeout_secs"].as_u64().unwrap_or(30),
                        ),
                        working_dir: ce["working_dir"].as_str().map(|s| s.to_string()),
                        env: Vec::new(),
                        sandbox,
                        interpreters: std::collections::HashMap::new(),
                    }
                } else {
                    gauss_core::code_execution::CodeExecutionConfig::all()
                };

                let unified = ce.get("unified").and_then(|v| v.as_bool()).unwrap_or(false);
                if unified {
                    builder = builder.code_execution_unified(ce_config);
                } else {
                    builder = builder.code_execution(ce_config);
                }
            }

            // Grounding (Gemini)
            if opts.get("grounding").and_then(|v| v.as_bool()).unwrap_or(false) {
                builder = builder.grounding(true);
            }
            // Native code execution (Gemini code interpreter)
            if opts.get("native_code_execution").and_then(|v| v.as_bool()).unwrap_or(false) {
                builder = builder.native_code_execution(true);
            }
            // Response modalities
            if let Some(modalities) = opts.get("response_modalities").and_then(|v| v.as_array()) {
                let mods: Vec<String> = modalities.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
                builder = builder.response_modalities(mods);
            }
        }

        let agent = builder.build();
        let output = agent
            .run(rust_msgs)
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("Agent error: {e}")))?;

        let agent_citations: Vec<serde_json::Value> = output.citations.iter().map(|c| json!({
            "type": c.citation_type,
            "cited_text": c.cited_text,
            "document_title": c.document_title,
            "start": c.start,
            "end": c.end,
        })).collect();

        let result = json!({
            "text": output.text,
            "thinking": output.thinking,
            "citations": agent_citations,
            "grounding_metadata": output.grounding_metadata,
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
        let entry: memory::MemoryEntry = serde_json::from_str(&entry_json).map_err(py_err)?;
        mem.store(entry).await.map_err(py_err)
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
            Some(j) => serde_json::from_str(&j).map_err(py_err)?,
            None => memory::RecallOptions::default(),
        };
        let entries = mem.recall(opts).await.map_err(py_err)?;
        serde_json::to_string(&entries).map_err(py_err)
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
            .map_err(py_err)
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
        let chunks: Vec<rag::Chunk> = serde_json::from_str(&chunks_json).map_err(py_err)?;
        store.upsert(chunks).await.map_err(py_err)
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
        let embedding: Vec<f32> = serde_json::from_str(&embedding_json).map_err(py_err)?;
        let results = store
            .search(&embedding, top_k as usize)
            .await
            .map_err(py_err)?;
        serde_json::to_string(&results).map_err(py_err)
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
    let a: Vec<f32> = serde_json::from_str(a_json).map_err(py_err)?;
    let b: Vec<f32> = serde_json::from_str(b_json).map_err(py_err)?;
    Ok(rag::cosine_similarity(&a, &b) as f64)
}

// ============ MCP ============

#[pyfunction]
fn create_mcp_server(name: &str, version_str: &str) -> u32 {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    mcp_servers()
        .lock()
        .expect("registry mutex poisoned")
        .insert(
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
    let mcp_tool: mcp::McpTool = serde_json::from_str(&tool_json).map_err(py_err)?;
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
    networks().lock().expect("registry mutex poisoned").insert(
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
            .map_err(py_err)?;
        serde_json::to_string(&output).map_err(py_err)
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
    let args: serde_json::Value = serde_json::from_str(&args_json).map_err(py_err)?;
    let req = mgr
        .lock()
        .unwrap()
        .request_approval(tool_name, args, 0, session_id)
        .map_err(py_err)?;
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
        Some(j) => Some(serde_json::from_str(&j).map_err(py_err)?),
        None => None,
    };
    mgr.lock()
        .unwrap()
        .approve(request_id, args)
        .map_err(py_err)?;
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
        .map_err(py_err)?;
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
    let pending = mgr
        .lock()
        .expect("registry mutex poisoned")
        .list_pending()
        .map_err(py_err)?;
    serde_json::to_string(&pending).map_err(py_err)
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
        let cp: hitl::Checkpoint = serde_json::from_str(&checkpoint_json).map_err(py_err)?;
        store.save(&cp).await.map_err(py_err)
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
        let cp = store.load(&checkpoint_id).await.map_err(py_err)?;
        serde_json::to_string(&cp).map_err(py_err)
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
        other => return Err(py_err(format!("Unknown scorer: {other}"))),
    };
    runner
        .lock()
        .expect("registry mutex poisoned")
        .add_scorer(scorer);
    Ok(())
}

#[pyfunction]
fn load_dataset_jsonl(jsonl: &str) -> PyResult<String> {
    let cases = eval::load_dataset_jsonl(jsonl).map_err(py_err)?;
    serde_json::to_string(&cases).map_err(py_err)
}

#[pyfunction]
fn load_dataset_json(json_str: &str) -> PyResult<String> {
    let cases = eval::load_dataset_json(json_str).map_err(py_err)?;
    serde_json::to_string(&cases).map_err(py_err)
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
    collectors()
        .lock()
        .expect("registry mutex poisoned")
        .insert(
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
    let span: telemetry::SpanRecord = serde_json::from_str(span_json).map_err(py_err)?;
    coll.lock()
        .expect("registry mutex poisoned")
        .record_span(span);
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
    let spans = coll.lock().expect("registry mutex poisoned").export_spans();
    serde_json::to_string(&spans).map_err(py_err)
}

#[pyfunction]
fn telemetry_export_metrics(handle: u32) -> PyResult<String> {
    let coll = collectors()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| py_err("TelemetryCollector not found"))?;
    let metrics = coll
        .lock()
        .expect("registry mutex poisoned")
        .export_metrics();
    serde_json::to_string(&metrics).map_err(py_err)
}

#[pyfunction]
fn telemetry_clear(handle: u32) -> PyResult<()> {
    let coll = collectors()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| py_err("TelemetryCollector not found"))?;
    coll.lock().expect("registry mutex poisoned").clear();
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

// ============ Guardrails ============

fn guardrail_chains() -> &'static Mutex<HashMap<u32, guardrail::GuardrailChain>> {
    use std::sync::OnceLock;
    static REG: OnceLock<Mutex<HashMap<u32, guardrail::GuardrailChain>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

#[pyfunction]
fn create_guardrail_chain() -> u32 {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    guardrail_chains()
        .lock()
        .unwrap()
        .insert(id, guardrail::GuardrailChain::new());
    id
}

#[pyfunction]
fn guardrail_chain_add_content_moderation(
    handle: u32,
    block_patterns: Vec<String>,
    warn_patterns: Vec<String>,
) -> PyResult<()> {
    let mut reg = guardrail_chains().lock().expect("registry mutex poisoned");
    let chain = reg
        .get_mut(&handle)
        .ok_or_else(|| py_err("GuardrailChain not found"))?;
    let mut g = guardrail::ContentModerationGuardrail::new();
    for p in block_patterns {
        g = g.block_pattern(&p, format!("Blocked: {p}"));
    }
    for p in warn_patterns {
        g = g.warn_pattern(&p, format!("Warning: {p}"));
    }
    chain.add(Arc::new(g));
    Ok(())
}

#[pyfunction]
fn guardrail_chain_add_pii_detection(handle: u32, action: String) -> PyResult<()> {
    let pii_action = match action.as_str() {
        "block" => guardrail::PiiAction::Block,
        "warn" => guardrail::PiiAction::Warn,
        "redact" => guardrail::PiiAction::Redact,
        _ => return Err(py_err("Invalid PII action: block|warn|redact")),
    };
    let mut reg = guardrail_chains().lock().expect("registry mutex poisoned");
    let chain = reg
        .get_mut(&handle)
        .ok_or_else(|| py_err("GuardrailChain not found"))?;
    chain.add(Arc::new(guardrail::PiiDetectionGuardrail::new(pii_action)));
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (handle, max_input=None, max_output=None))]
fn guardrail_chain_add_token_limit(
    handle: u32,
    max_input: Option<u32>,
    max_output: Option<u32>,
) -> PyResult<()> {
    let mut g = guardrail::TokenLimitGuardrail::new();
    if let Some(m) = max_input {
        g = g.max_input(m as usize);
    }
    if let Some(m) = max_output {
        g = g.max_output(m as usize);
    }
    let mut reg = guardrail_chains().lock().expect("registry mutex poisoned");
    let chain = reg
        .get_mut(&handle)
        .ok_or_else(|| py_err("GuardrailChain not found"))?;
    chain.add(Arc::new(g));
    Ok(())
}

#[pyfunction]
fn guardrail_chain_add_regex_filter(
    handle: u32,
    block_rules: Vec<String>,
    warn_rules: Vec<String>,
) -> PyResult<()> {
    let mut g = guardrail::RegexFilterGuardrail::new();
    for r in block_rules {
        g = g.block(&r, format!("Blocked by regex: {r}"));
    }
    for r in warn_rules {
        g = g.warn(&r, format!("Warning by regex: {r}"));
    }
    let mut reg = guardrail_chains().lock().expect("registry mutex poisoned");
    let chain = reg
        .get_mut(&handle)
        .ok_or_else(|| py_err("GuardrailChain not found"))?;
    chain.add(Arc::new(g));
    Ok(())
}

#[pyfunction]
fn guardrail_chain_add_schema(handle: u32, schema_json: String) -> PyResult<()> {
    let schema: serde_json::Value = serde_json::from_str(&schema_json)
        .map_err(|e| py_err(format!("Invalid JSON schema: {e}")))?;
    let mut reg = guardrail_chains().lock().expect("registry mutex poisoned");
    let chain = reg
        .get_mut(&handle)
        .ok_or_else(|| py_err("GuardrailChain not found"))?;
    chain.add(Arc::new(guardrail::SchemaGuardrail::new(schema)));
    Ok(())
}

#[pyfunction]
fn guardrail_chain_list(handle: u32) -> PyResult<Vec<String>> {
    let reg = guardrail_chains().lock().expect("registry mutex poisoned");
    let chain = reg
        .get(&handle)
        .ok_or_else(|| py_err("GuardrailChain not found"))?;
    Ok(chain.list().into_iter().map(String::from).collect())
}

#[pyfunction]
fn destroy_guardrail_chain(handle: u32) -> PyResult<()> {
    guardrail_chains()
        .lock()
        .unwrap()
        .remove(&handle)
        .ok_or_else(|| py_err("GuardrailChain not found"))?;
    Ok(())
}

// ============ Resilience ============

#[pyfunction]
fn create_fallback_provider(provider_handles: Vec<u32>) -> PyResult<u32> {
    let prov_reg = providers().lock().expect("registry mutex poisoned");
    let mut providers_vec: Vec<Arc<dyn Provider>> = Vec::new();
    for h in provider_handles {
        let p = prov_reg
            .get(&h)
            .ok_or_else(|| py_err(format!("Provider {h} not found")))?
            .clone();
        providers_vec.push(p);
    }
    drop(prov_reg);

    let fallback = Arc::new(resilience::FallbackProvider::new(providers_vec));
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    providers()
        .lock()
        .expect("registry mutex poisoned")
        .insert(id, fallback);
    Ok(id)
}

#[pyfunction]
#[pyo3(signature = (provider_handle, failure_threshold=None, recovery_timeout_ms=None))]
fn create_circuit_breaker(
    provider_handle: u32,
    failure_threshold: Option<u32>,
    recovery_timeout_ms: Option<u32>,
) -> PyResult<u32> {
    let inner = providers()
        .lock()
        .unwrap()
        .get(&provider_handle)
        .ok_or_else(|| py_err("Provider not found"))?
        .clone();

    let config = resilience::CircuitBreakerConfig {
        failure_threshold: failure_threshold.unwrap_or(5),
        recovery_timeout_ms: recovery_timeout_ms.map(|v| v as u64).unwrap_or(30_000),
        success_threshold: 1,
    };

    let cb = Arc::new(resilience::CircuitBreaker::new(inner, config));
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    providers()
        .lock()
        .expect("registry mutex poisoned")
        .insert(id, cb);
    Ok(id)
}

#[pyfunction]
#[pyo3(signature = (primary_handle, fallback_handles, enable_circuit_breaker=None))]
fn create_resilient_provider(
    primary_handle: u32,
    fallback_handles: Vec<u32>,
    enable_circuit_breaker: Option<bool>,
) -> PyResult<u32> {
    let prov_reg = providers().lock().expect("registry mutex poisoned");
    let primary = prov_reg
        .get(&primary_handle)
        .ok_or_else(|| py_err("Primary provider not found"))?
        .clone();

    let mut builder = resilience::ResilientProviderBuilder::new(primary);
    builder = builder.retry(RetryConfig::default());

    if enable_circuit_breaker.unwrap_or(false) {
        builder = builder.circuit_breaker(resilience::CircuitBreakerConfig::default());
    }

    for h in &fallback_handles {
        let fb = prov_reg
            .get(h)
            .ok_or_else(|| py_err(format!("Fallback provider {h} not found")))?
            .clone();
        builder = builder.fallback(fb);
    }
    drop(prov_reg);

    let provider = builder.build();
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    providers()
        .lock()
        .expect("registry mutex poisoned")
        .insert(id, provider);
    Ok(id)
}

// ============ Stream Transform ============

#[pyfunction]
fn py_parse_partial_json(text: String) -> Option<String> {
    stream_transform::parse_partial_json(&text).map(|v| v.to_string())
}

// ============ Plugin System ============

fn plugin_registries() -> &'static Mutex<HashMap<u32, plugin::PluginRegistry>> {
    use std::sync::OnceLock;
    static REG: OnceLock<Mutex<HashMap<u32, plugin::PluginRegistry>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

#[pyfunction]
fn create_plugin_registry() -> u32 {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    plugin_registries()
        .lock()
        .unwrap()
        .insert(id, plugin::PluginRegistry::new());
    id
}

#[pyfunction]
fn plugin_registry_add_telemetry(handle: u32) -> PyResult<()> {
    let mut reg = plugin_registries().lock().expect("registry mutex poisoned");
    let registry = reg
        .get_mut(&handle)
        .ok_or_else(|| py_err("PluginRegistry not found"))?;
    registry.register(Arc::new(plugin::TelemetryPlugin));
    Ok(())
}

#[pyfunction]
fn plugin_registry_add_memory(handle: u32) -> PyResult<()> {
    let mut reg = plugin_registries().lock().expect("registry mutex poisoned");
    let registry = reg
        .get_mut(&handle)
        .ok_or_else(|| py_err("PluginRegistry not found"))?;
    registry.register(Arc::new(plugin::MemoryPlugin));
    Ok(())
}

#[pyfunction]
fn plugin_registry_list(handle: u32) -> PyResult<Vec<String>> {
    let reg = plugin_registries().lock().expect("registry mutex poisoned");
    let registry = reg
        .get(&handle)
        .ok_or_else(|| py_err("PluginRegistry not found"))?;
    Ok(registry.list().into_iter().map(String::from).collect())
}

#[pyfunction]
fn plugin_registry_emit(handle: u32, event_json: String) -> PyResult<()> {
    let event: plugin::GaussEvent = serde_json::from_str(&event_json)
        .map_err(|e| py_err(format!("Invalid event JSON: {e}")))?;
    let reg = plugin_registries().lock().expect("registry mutex poisoned");
    let registry = reg
        .get(&handle)
        .ok_or_else(|| py_err("PluginRegistry not found"))?;
    registry.emit(&event);
    Ok(())
}

#[pyfunction]
fn destroy_plugin_registry(handle: u32) -> PyResult<()> {
    plugin_registries()
        .lock()
        .unwrap()
        .remove(&handle)
        .ok_or_else(|| py_err("PluginRegistry not found"))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tool Validator (patterns module)
// ---------------------------------------------------------------------------

use gauss_core::patterns::{CoercionStrategy, ToolValidator as RustToolValidator};

fn tool_validators() -> &'static Mutex<HashMap<u32, RustToolValidator>> {
    static REG: OnceLock<Mutex<HashMap<u32, RustToolValidator>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

#[pyfunction]
#[pyo3(signature = (strategies=None))]
fn create_tool_validator(strategies: Option<Vec<String>>) -> u32 {
    let validator = match strategies {
        Some(strats) => {
            let parsed: Vec<CoercionStrategy> = strats
                .iter()
                .filter_map(|s| match s.as_str() {
                    "null_to_default" => Some(CoercionStrategy::NullToDefault),
                    "type_cast" => Some(CoercionStrategy::TypeCast),
                    "json_parse" => Some(CoercionStrategy::JsonParse),
                    "strip_null" => Some(CoercionStrategy::StripNull),
                    _ => None,
                })
                .collect();
            RustToolValidator::with_strategies(parsed)
        }
        None => RustToolValidator::new(),
    };
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    tool_validators()
        .lock()
        .expect("registry mutex poisoned")
        .insert(id, validator);
    id
}

#[pyfunction]
fn tool_validator_validate(handle: u32, input: String, schema: String) -> PyResult<String> {
    let reg = tool_validators().lock().expect("registry mutex poisoned");
    let validator = reg
        .get(&handle)
        .ok_or_else(|| py_err("ToolValidator not found"))?;
    let input_val: serde_json::Value =
        serde_json::from_str(&input).map_err(|e| py_err(format!("{e}")))?;
    let schema_val: serde_json::Value =
        serde_json::from_str(&schema).map_err(|e| py_err(format!("{e}")))?;
    let result = validator
        .validate(input_val, &schema_val)
        .map_err(|e| py_err(format!("{e}")))?;
    serde_json::to_string(&result).map_err(|e| py_err(format!("{e}")))
}

#[pyfunction]
fn destroy_tool_validator(handle: u32) -> PyResult<()> {
    tool_validators()
        .lock()
        .unwrap()
        .remove(&handle)
        .ok_or_else(|| py_err("ToolValidator not found"))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Agent Config (config module)
// ---------------------------------------------------------------------------

#[pyfunction]
fn agent_config_from_json(json_str: String) -> PyResult<String> {
    let config = gauss_core::config::AgentConfig::from_json(&json_str)
        .map_err(|e| py_err(format!("{e}")))?;
    config.to_json().map_err(|e| py_err(format!("{e}")))
}

#[pyfunction]
fn agent_config_resolve_env(value: String) -> String {
    gauss_core::config::resolve_env(&value)
}

// ============ Graph ============

struct GraphNodeDef {
    id: String,
    agent_name: String,
    provider_handle: u32,
    instructions: Option<String>,
    tools: Vec<(String, String, Option<serde_json::Value>)>,
}

struct GraphForkDef {
    id: String,
    agents: Vec<GraphNodeDef>,
    consensus: String,
}

enum GraphNodeDefKind {
    Agent(GraphNodeDef),
    Fork(GraphForkDef),
}

struct GraphState {
    nodes: Vec<GraphNodeDefKind>,
    edges: Vec<(String, String)>,
}

fn graphs() -> &'static Mutex<HashMap<u32, GraphState>> {
    static R: OnceLock<Mutex<HashMap<u32, GraphState>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

#[pyfunction]
fn create_graph() -> u32 {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    graphs().lock().expect("registry mutex poisoned").insert(
        id,
        GraphState {
            nodes: vec![],
            edges: vec![],
        },
    );
    id
}

#[pyfunction]
#[pyo3(signature = (handle, node_id, agent_name, provider_handle, instructions=None, tools_json=None))]
fn graph_add_node(
    handle: u32,
    node_id: String,
    agent_name: String,
    provider_handle: u32,
    instructions: Option<String>,
    tools_json: Option<String>,
) -> PyResult<()> {
    let tools: Vec<(String, String, Option<serde_json::Value>)> = match tools_json {
        Some(j) => {
            let parsed: Vec<serde_json::Value> = serde_json::from_str(&j).map_err(py_err)?;
            parsed
                .into_iter()
                .map(|t| {
                    let name = t["name"].as_str().unwrap_or("").to_string();
                    let desc = t["description"].as_str().unwrap_or("").to_string();
                    let params = t.get("parameters").cloned();
                    (name, desc, params)
                })
                .collect()
        }
        None => vec![],
    };
    let mut g = graphs().lock().expect("registry mutex poisoned");
    let state = g
        .get_mut(&handle)
        .ok_or_else(|| py_err("Graph not found"))?;
    state.nodes.push(GraphNodeDefKind::Agent(GraphNodeDef {
        id: node_id,
        agent_name,
        provider_handle,
        instructions,
        tools,
    }));
    Ok(())
}

#[pyfunction]
fn graph_add_edge(handle: u32, from: String, to: String) -> PyResult<()> {
    let mut g = graphs().lock().expect("registry mutex poisoned");
    let state = g
        .get_mut(&handle)
        .ok_or_else(|| py_err("Graph not found"))?;
    state.edges.push((from, to));
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (handle, node_id, agents_json, consensus))]
fn graph_add_fork_node(
    handle: u32,
    node_id: String,
    agents_json: String,
    consensus: String,
) -> PyResult<()> {
    let parsed: Vec<serde_json::Value> = serde_json::from_str(&agents_json).map_err(py_err)?;
    let agents = parsed
        .into_iter()
        .map(|a| {
            let name = a["agent_name"].as_str().unwrap_or("").to_string();
            GraphNodeDef {
                id: name.clone(),
                agent_name: name,
                provider_handle: a["provider_handle"].as_u64().unwrap_or(0) as u32,
                instructions: a["instructions"].as_str().map(|s| s.to_string()),
                tools: vec![],
            }
        })
        .collect();
    let mut g = graphs().lock().expect("registry mutex poisoned");
    let state = g
        .get_mut(&handle)
        .ok_or_else(|| py_err("Graph not found"))?;
    state.nodes.push(GraphNodeDefKind::Fork(GraphForkDef {
        id: node_id,
        agents,
        consensus,
    }));
    Ok(())
}

#[pyfunction]
fn graph_run<'py>(
    py: Python<'py>,
    handle: u32,
    prompt: String,
) -> PyResult<Bound<'py, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        enum NodeSnapshot {
            Agent {
                id: String,
                agent_name: String,
                provider_handle: u32,
                instructions: Option<String>,
                tools: Vec<(String, String, Option<serde_json::Value>)>,
            },
            Fork {
                id: String,
                agents: Vec<(String, u32, Option<String>)>,
                consensus: String,
            },
        }

        let (snapshots, edges) = {
            let g = graphs().lock().expect("registry mutex poisoned");
            let s = g.get(&handle).ok_or_else(|| py_err("Graph not found"))?;
            let snapshots: Vec<NodeSnapshot> = s
                .nodes
                .iter()
                .map(|n| match n {
                    GraphNodeDefKind::Agent(a) => NodeSnapshot::Agent {
                        id: a.id.clone(),
                        agent_name: a.agent_name.clone(),
                        provider_handle: a.provider_handle,
                        instructions: a.instructions.clone(),
                        tools: a.tools.clone(),
                    },
                    GraphNodeDefKind::Fork(f) => NodeSnapshot::Fork {
                        id: f.id.clone(),
                        agents: f
                            .agents
                            .iter()
                            .map(|a| {
                                (
                                    a.agent_name.clone(),
                                    a.provider_handle,
                                    a.instructions.clone(),
                                )
                            })
                            .collect(),
                        consensus: f.consensus.clone(),
                    },
                })
                .collect();
            Ok::<_, PyErr>((snapshots, s.edges.clone()))
        }?;

        let mut builder = gauss_core::Graph::builder();

        for snap in &snapshots {
            match snap {
                NodeSnapshot::Agent {
                    id,
                    agent_name,
                    provider_handle,
                    instructions,
                    tools,
                } => {
                    let provider = get_provider(*provider_handle)?;
                    let mut ab = RustAgent::builder(agent_name.clone(), provider);
                    if let Some(instr) = instructions {
                        ab = ab.instructions(instr.clone());
                    }
                    for (name, desc, params) in tools {
                        let mut tb = gauss_core::tool::Tool::builder(name, desc);
                        if let Some(p) = params {
                            tb = tb.parameters_json(p.clone());
                        }
                        ab = ab.tool(tb.build());
                    }
                    let agent = ab.build();
                    let nid = id.clone();
                    builder = builder.node(nid, agent, |_deps| {
                        vec![RustMessage::user("Continue based on prior context.")]
                    });
                }
                NodeSnapshot::Fork {
                    id,
                    agents,
                    consensus,
                } => {
                    let mut fork_agents: Vec<(String, gauss_core::Agent)> = Vec::new();
                    for (name, prov_handle, instructions) in agents {
                        let provider = get_provider(*prov_handle)?;
                        let mut ab = RustAgent::builder(name.clone(), provider);
                        if let Some(instr) = instructions {
                            ab = ab.instructions(instr.clone());
                        }
                        fork_agents.push((name.clone(), ab.build()));
                    }
                    let strategy = match consensus.as_str() {
                        "first" => gauss_core::graph::ConsensusStrategy::First,
                        _ => gauss_core::graph::ConsensusStrategy::Concat,
                    };
                    builder = builder.fork(id.clone(), fork_agents, strategy);
                }
            }
        }

        for (from, to) in &edges {
            builder = builder.edge(from.clone(), to.clone());
        }

        let graph = builder.build();
        let result = graph.run(prompt).await.map_err(py_err)?;

        let outputs: serde_json::Value = result
            .outputs
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    json!({ "node_id": v.node_id, "text": v.text, "data": v.data }),
                )
            })
            .collect::<serde_json::Map<String, serde_json::Value>>()
            .into();

        let result_json = serde_json::to_string(&json!({
            "outputs": outputs,
            "final_text": result.final_output.map(|o| o.text),
        }))
        .map_err(py_err)?;

        Ok(result_json)
    })
}

#[pyfunction]
fn destroy_graph(handle: u32) -> PyResult<()> {
    graphs()
        .lock()
        .unwrap()
        .remove(&handle)
        .ok_or_else(|| py_err("Graph not found"))?;
    Ok(())
}

// ============ Workflow ============

struct WorkflowStepDef {
    id: String,
    agent_name: String,
    provider_handle: u32,
    instructions: Option<String>,
    tools: Vec<(String, String, Option<serde_json::Value>)>,
}

struct WorkflowState {
    steps: Vec<WorkflowStepDef>,
    dependencies: Vec<(String, String)>,
}

fn workflow_states() -> &'static Mutex<HashMap<u32, WorkflowState>> {
    static R: OnceLock<Mutex<HashMap<u32, WorkflowState>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

#[pyfunction]
fn create_workflow() -> u32 {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    workflow_states()
        .lock()
        .expect("registry mutex poisoned")
        .insert(
            id,
            WorkflowState {
                steps: vec![],
                dependencies: vec![],
            },
        );
    id
}

#[pyfunction]
#[pyo3(signature = (handle, step_id, agent_name, provider_handle, instructions=None, tools_json=None))]
fn workflow_add_step(
    handle: u32,
    step_id: String,
    agent_name: String,
    provider_handle: u32,
    instructions: Option<String>,
    tools_json: Option<String>,
) -> PyResult<()> {
    let tools: Vec<(String, String, Option<serde_json::Value>)> = match tools_json {
        Some(j) => {
            let parsed: Vec<serde_json::Value> = serde_json::from_str(&j).map_err(py_err)?;
            parsed
                .into_iter()
                .map(|t| {
                    let name = t["name"].as_str().unwrap_or("").to_string();
                    let desc = t["description"].as_str().unwrap_or("").to_string();
                    let params = t.get("parameters").cloned();
                    (name, desc, params)
                })
                .collect()
        }
        None => vec![],
    };
    let mut w = workflow_states().lock().expect("registry mutex poisoned");
    let state = w
        .get_mut(&handle)
        .ok_or_else(|| py_err("Workflow not found"))?;
    state.steps.push(WorkflowStepDef {
        id: step_id,
        agent_name,
        provider_handle,
        instructions,
        tools,
    });
    Ok(())
}

#[pyfunction]
fn workflow_add_dependency(handle: u32, step_id: String, depends_on: String) -> PyResult<()> {
    let mut w = workflow_states().lock().expect("registry mutex poisoned");
    let state = w
        .get_mut(&handle)
        .ok_or_else(|| py_err("Workflow not found"))?;
    state.dependencies.push((step_id, depends_on));
    Ok(())
}

#[pyfunction]
fn workflow_run<'py>(
    py: Python<'py>,
    handle: u32,
    prompt: String,
) -> PyResult<Bound<'py, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let state = {
            let w = workflow_states().lock().expect("registry mutex poisoned");
            let s = w.get(&handle).ok_or_else(|| py_err("Workflow not found"))?;
            let steps: Vec<_> = s
                .steps
                .iter()
                .map(|st| {
                    (
                        st.id.clone(),
                        st.agent_name.clone(),
                        st.provider_handle,
                        st.instructions.clone(),
                        st.tools.clone(),
                    )
                })
                .collect();
            let deps = s.dependencies.clone();
            Ok::<_, PyErr>((steps, deps))
        }?;

        let (step_defs, deps) = state;
        let mut builder = gauss_core::Workflow::builder();

        for (step_id, agent_name, prov_handle, instructions, tools) in &step_defs {
            let provider = get_provider(*prov_handle)?;
            let mut agent_builder = RustAgent::builder(agent_name.clone(), provider);
            if let Some(instr) = instructions {
                agent_builder = agent_builder.instructions(instr.clone());
            }
            for (name, desc, params) in tools {
                let mut tb = gauss_core::tool::Tool::builder(name, desc);
                if let Some(p) = params {
                    tb = tb.parameters_json(p.clone());
                }
                agent_builder = agent_builder.tool(tb.build());
            }
            let agent = agent_builder.build();
            builder = builder.agent_step(step_id.clone(), agent, |_completed| {
                vec![RustMessage::user("Continue.")]
            });
        }

        for (step_id, depends_on) in &deps {
            builder = builder.dependency(step_id.clone(), depends_on.clone());
        }

        let workflow = builder.build();
        let initial = vec![RustMessage::user(prompt)];
        let result = workflow.run(initial).await.map_err(py_err)?;

        let outputs: serde_json::Value = result
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    json!({ "step_id": v.step_id, "text": v.text, "data": v.data }),
                )
            })
            .collect::<serde_json::Map<String, serde_json::Value>>()
            .into();

        let result_json = serde_json::to_string(&json!({ "steps": outputs })).map_err(py_err)?;
        Ok(result_json)
    })
}

#[pyfunction]
fn destroy_workflow(handle: u32) -> PyResult<()> {
    workflow_states()
        .lock()
        .unwrap()
        .remove(&handle)
        .ok_or_else(|| py_err("Workflow not found"))?;
    Ok(())
}

// ============ Middleware ============

fn middleware_chains()
-> &'static Mutex<HashMap<u32, Arc<Mutex<gauss_core::middleware::MiddlewareChain>>>> {
    static R: OnceLock<Mutex<HashMap<u32, Arc<Mutex<gauss_core::middleware::MiddlewareChain>>>>> =
        OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

#[pyfunction]
fn create_middleware_chain() -> u32 {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    middleware_chains().lock().unwrap().insert(
        id,
        Arc::new(Mutex::new(gauss_core::middleware::MiddlewareChain::new())),
    );
    id
}

#[pyfunction]
fn middleware_use_logging(handle: u32) -> PyResult<()> {
    let reg = middleware_chains();
    let guard = reg.lock().expect("registry mutex poisoned");
    let chain = guard
        .get(&handle)
        .ok_or_else(|| py_err("MiddlewareChain not found"))?;
    chain
        .lock()
        .unwrap()
        .use_middleware(Arc::new(gauss_core::middleware::LoggingMiddleware));
    Ok(())
}

#[pyfunction]
fn middleware_use_caching(handle: u32, ttl_ms: u32) -> PyResult<()> {
    let reg = middleware_chains();
    let guard = reg.lock().expect("registry mutex poisoned");
    let chain = guard
        .get(&handle)
        .ok_or_else(|| py_err("MiddlewareChain not found"))?;
    chain
        .lock()
        .unwrap()
        .use_middleware(Arc::new(gauss_core::middleware::CachingMiddleware::new(
            ttl_ms as u64,
        )));
    Ok(())
}

#[pyfunction]
fn destroy_middleware_chain(handle: u32) -> PyResult<()> {
    middleware_chains()
        .lock()
        .unwrap()
        .remove(&handle)
        .ok_or_else(|| py_err("MiddlewareChain not found"))?;
    Ok(())
}

// ============ Memory Stats ============

#[pyfunction]
fn memory_stats<'py>(py: Python<'py>, handle: u32) -> PyResult<Bound<'py, pyo3::types::PyAny>> {
    let mem = memories()
        .lock()
        .expect("registry mutex poisoned")
        .get(&handle)
        .cloned()
        .ok_or_else(|| py_err("Memory not found"))?;
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let stats = mem.stats().await.map_err(py_err)?;
        serde_json::to_string(&stats).map_err(py_err)
    })
}

// ============ Network Agent Cards ============

#[pyfunction]
fn network_agent_cards(handle: u32) -> PyResult<String> {
    let net = networks()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| py_err("Network not found"))?;
    let guard = net.blocking_lock();
    let cards = guard.agent_cards();
    serde_json::to_string(&cards).map_err(py_err)
}

// ============ Checkpoint Load Latest ============

#[pyfunction]
fn checkpoint_load_latest<'py>(
    py: Python<'py>,
    handle: u32,
    session_id: String,
) -> PyResult<Bound<'py, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let store = checkpoints()
            .lock()
            .unwrap()
            .get(&handle)
            .cloned()
            .ok_or_else(|| py_err("CheckpointStore not found"))?;
        use gauss_core::hitl::CheckpointStore;
        match store.load_latest(&session_id).await {
            Ok(Some(cp)) => serde_json::to_string(&cp).map_err(py_err),
            Ok(None) => Ok("null".to_string()),
            Err(e) => Err(py_err(e)),
        }
    })
}

// ============ Team ============

#[derive(Clone)]
struct TeamAgentDef {
    name: String,
    provider_handle: u32,
    instructions: Option<String>,
}

#[derive(Clone)]
struct TeamState {
    name: String,
    strategy: String,
    agents: Vec<TeamAgentDef>,
}

fn teams() -> &'static Mutex<HashMap<u32, TeamState>> {
    static R: OnceLock<Mutex<HashMap<u32, TeamState>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

#[pyfunction]
fn create_team(name: String) -> u32 {
    static COUNTER: AtomicU32 = AtomicU32::new(1);
    let handle = COUNTER.fetch_add(1, Ordering::Relaxed);
    teams().lock().unwrap().insert(
        handle,
        TeamState {
            name,
            strategy: "sequential".to_string(),
            agents: Vec::new(),
        },
    );
    handle
}

#[pyfunction]
#[pyo3(signature = (handle, agent_name, provider_handle, instructions=None))]
fn team_add_agent(
    handle: u32,
    agent_name: String,
    provider_handle: u32,
    instructions: Option<String>,
) -> PyResult<()> {
    let mut t = teams().lock().unwrap();
    let state = t
        .get_mut(&handle)
        .ok_or_else(|| py_err("Team not found"))?;
    state.agents.push(TeamAgentDef {
        name: agent_name,
        provider_handle,
        instructions,
    });
    Ok(())
}

#[pyfunction]
fn team_set_strategy(handle: u32, strategy: String) -> PyResult<()> {
    let mut t = teams().lock().unwrap();
    let state = t
        .get_mut(&handle)
        .ok_or_else(|| py_err("Team not found"))?;
    match strategy.as_str() {
        "sequential" | "parallel" => {
            state.strategy = strategy;
            Ok(())
        }
        _ => Err(py_err(format!(
            "Unknown strategy: {strategy}. Use 'sequential' or 'parallel'"
        ))),
    }
}

#[pyfunction]
fn team_run<'py>(
    py: Python<'py>,
    handle: u32,
    messages_json: String,
) -> PyResult<Bound<'py, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let state = teams()
            .lock()
            .unwrap()
            .get(&handle)
            .cloned()
            .ok_or_else(|| py_err("Team not found"))?;

        let strategy = match state.strategy.as_str() {
            "parallel" => team::Strategy::Parallel,
            _ => team::Strategy::Sequential,
        };

        let mut builder = team::Team::builder(&state.name).strategy(strategy);

        for agent_def in &state.agents {
            let provider = get_provider(agent_def.provider_handle)?;
            let mut ab = RustAgent::builder(&agent_def.name, provider);
            if let Some(ref instr) = agent_def.instructions {
                ab = ab.instructions(instr.clone());
            }
            builder = builder.agent(ab.build());
        }

        let team_instance = builder.build();

        let messages: Vec<RustMessage> = serde_json::from_str(&messages_json)
            .map_err(|e| py_err(format!("Invalid messages JSON: {e}")))?;

        let output = team_instance
            .run(messages)
            .await
            .map_err(|e| py_err(format!("Team run error: {e}")))?;

        let results: Vec<serde_json::Value> = output
            .results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "text": r.text,
                    "steps": r.steps,
                    "inputTokens": r.usage.input_tokens,
                    "outputTokens": r.usage.output_tokens,
                })
            })
            .collect();

        serde_json::to_string(&serde_json::json!({
            "finalText": output.final_text,
            "results": results,
        }))
        .map_err(|e| py_err(format!("Serialize error: {e}")))
    })
}

#[pyfunction]
fn destroy_team(handle: u32) -> PyResult<()> {
    teams()
        .lock()
        .unwrap()
        .remove(&handle)
        .ok_or_else(|| py_err("Team not found"))?;
    Ok(())
}

// ============ Code Execution (PTC) ============

#[pyfunction]
#[pyo3(signature = (language, code, timeout_secs=None, working_dir=None, sandbox=None))]
fn execute_code(
    py: Python<'_>,
    language: String,
    code: String,
    timeout_secs: Option<u64>,
    working_dir: Option<String>,
    sandbox: Option<String>,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let sandbox_config = match sandbox.as_deref() {
            Some("strict") => gauss_core::code_execution::SandboxConfig::strict(),
            Some("permissive") => gauss_core::code_execution::SandboxConfig::permissive(),
            _ => gauss_core::code_execution::SandboxConfig::default(),
        };

        let config = gauss_core::code_execution::CodeExecutionConfig {
            python: language == "python",
            javascript: language == "javascript",
            bash: language == "bash",
            timeout: std::time::Duration::from_secs(timeout_secs.unwrap_or(30)),
            working_dir,
            env: Vec::new(),
            sandbox: sandbox_config,
            interpreters: std::collections::HashMap::new(),
        };

        let orch = gauss_core::code_execution::CodeExecutionOrchestrator::new(config);
        let result = orch
            .execute(&language, &code)
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("Code execution error: {e}")))?;

        let result_json = json!({
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "runtime": result.runtime,
            "success": result.success(),
        });

        serde_json::to_string(&result_json)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {e}")))
    })
}

#[pyfunction]
fn available_runtimes(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let config = gauss_core::code_execution::CodeExecutionConfig::all();
        let orch = gauss_core::code_execution::CodeExecutionOrchestrator::new(config);
        let runtimes = orch.available_runtimes().await;
        serde_json::to_string(&runtimes)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {e}")))
    })
}

/// Generate images using a provider's image generation API.
#[pyfunction]
#[pyo3(signature = (provider_handle, prompt, model=None, size=None, quality=None, style=None, aspect_ratio=None, n=None, response_format=None))]
fn generate_image(
    py: Python<'_>,
    provider_handle: u32,
    prompt: String,
    model: Option<String>,
    size: Option<String>,
    quality: Option<String>,
    style: Option<String>,
    aspect_ratio: Option<String>,
    n: Option<u32>,
    response_format: Option<String>,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let provider = get_provider(provider_handle)?;

        let config = gauss_core::ImageGenerationConfig {
            model,
            size,
            quality,
            style,
            aspect_ratio,
            n,
            response_format,
        };

        let result = provider
            .generate_image(&prompt, &config)
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("Image generation error: {e}")))?;

        serde_json::to_string(&result)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {e}")))
    })
}

/// Gauss Core Python module.
#[pymodule]
#[pyo3(name = "_native")]
fn gauss_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(create_provider, m)?)?;
    m.add_function(wrap_pyfunction!(destroy_provider, m)?)?;
    m.add_function(wrap_pyfunction!(get_provider_capabilities, m)?)?;
    m.add_function(wrap_pyfunction!(generate, m)?)?;
    m.add_function(wrap_pyfunction!(stream_generate, m)?)?;
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
    // Guardrails
    m.add_function(wrap_pyfunction!(create_guardrail_chain, m)?)?;
    m.add_function(wrap_pyfunction!(guardrail_chain_add_content_moderation, m)?)?;
    m.add_function(wrap_pyfunction!(guardrail_chain_add_pii_detection, m)?)?;
    m.add_function(wrap_pyfunction!(guardrail_chain_add_token_limit, m)?)?;
    m.add_function(wrap_pyfunction!(guardrail_chain_add_regex_filter, m)?)?;
    m.add_function(wrap_pyfunction!(guardrail_chain_add_schema, m)?)?;
    m.add_function(wrap_pyfunction!(guardrail_chain_list, m)?)?;
    m.add_function(wrap_pyfunction!(destroy_guardrail_chain, m)?)?;
    // Resilience
    m.add_function(wrap_pyfunction!(create_fallback_provider, m)?)?;
    m.add_function(wrap_pyfunction!(create_circuit_breaker, m)?)?;
    m.add_function(wrap_pyfunction!(create_resilient_provider, m)?)?;
    // Stream Transform
    m.add_function(wrap_pyfunction!(py_parse_partial_json, m)?)?;
    // Plugin
    m.add_function(wrap_pyfunction!(create_plugin_registry, m)?)?;
    m.add_function(wrap_pyfunction!(plugin_registry_add_telemetry, m)?)?;
    m.add_function(wrap_pyfunction!(plugin_registry_add_memory, m)?)?;
    m.add_function(wrap_pyfunction!(plugin_registry_list, m)?)?;
    m.add_function(wrap_pyfunction!(plugin_registry_emit, m)?)?;
    m.add_function(wrap_pyfunction!(destroy_plugin_registry, m)?)?;
    // Patterns
    m.add_function(wrap_pyfunction!(create_tool_validator, m)?)?;
    m.add_function(wrap_pyfunction!(tool_validator_validate, m)?)?;
    m.add_function(wrap_pyfunction!(destroy_tool_validator, m)?)?;
    // Config
    m.add_function(wrap_pyfunction!(agent_config_from_json, m)?)?;
    m.add_function(wrap_pyfunction!(agent_config_resolve_env, m)?)?;
    // Graph
    m.add_function(wrap_pyfunction!(create_graph, m)?)?;
    m.add_function(wrap_pyfunction!(graph_add_node, m)?)?;
    m.add_function(wrap_pyfunction!(graph_add_edge, m)?)?;
    m.add_function(wrap_pyfunction!(graph_add_fork_node, m)?)?;
    m.add_function(wrap_pyfunction!(graph_run, m)?)?;
    m.add_function(wrap_pyfunction!(destroy_graph, m)?)?;
    // Workflow
    m.add_function(wrap_pyfunction!(create_workflow, m)?)?;
    m.add_function(wrap_pyfunction!(workflow_add_step, m)?)?;
    m.add_function(wrap_pyfunction!(workflow_add_dependency, m)?)?;
    m.add_function(wrap_pyfunction!(workflow_run, m)?)?;
    m.add_function(wrap_pyfunction!(destroy_workflow, m)?)?;
    // Middleware
    m.add_function(wrap_pyfunction!(create_middleware_chain, m)?)?;
    m.add_function(wrap_pyfunction!(middleware_use_logging, m)?)?;
    m.add_function(wrap_pyfunction!(middleware_use_caching, m)?)?;
    m.add_function(wrap_pyfunction!(destroy_middleware_chain, m)?)?;
    // Additional parity functions
    m.add_function(wrap_pyfunction!(memory_stats, m)?)?;
    m.add_function(wrap_pyfunction!(network_agent_cards, m)?)?;
    m.add_function(wrap_pyfunction!(checkpoint_load_latest, m)?)?;
    // Team
    m.add_function(wrap_pyfunction!(create_team, m)?)?;
    m.add_function(wrap_pyfunction!(team_add_agent, m)?)?;
    m.add_function(wrap_pyfunction!(team_set_strategy, m)?)?;
    m.add_function(wrap_pyfunction!(team_run, m)?)?;
    m.add_function(wrap_pyfunction!(destroy_team, m)?)?;
    // Code Execution (PTC)
    m.add_function(wrap_pyfunction!(execute_code, m)?)?;
    m.add_function(wrap_pyfunction!(available_runtimes, m)?)?;
    // Image Generation
    m.add_function(wrap_pyfunction!(generate_image, m)?)?;
    Ok(())
}
