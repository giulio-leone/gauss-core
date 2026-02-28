// WASM crate: Shared = Rc on wasm32, Arc on native. RefCell is fine for single-threaded WASM.
#![allow(clippy::arc_with_non_send_sync)]
#![allow(clippy::await_holding_refcell_ref)]

use futures::StreamExt;
use gauss_core::Shared;
use gauss_core::agent::{Agent as RustAgent, StopCondition};
use gauss_core::context;
use gauss_core::eval;
use gauss_core::hitl;
use gauss_core::mcp;
use gauss_core::memory;
use gauss_core::message::Message as RustMessage;
use gauss_core::provider::anthropic::AnthropicProvider;
use gauss_core::provider::deepseek::DeepSeekProvider;
use gauss_core::provider::google::GoogleProvider;
use gauss_core::provider::groq::GroqProvider;
use gauss_core::provider::ollama::OllamaProvider;
use gauss_core::provider::openai::OpenAiProvider;
use gauss_core::provider::retry::{RetryConfig, RetryProvider};
use gauss_core::provider::{GenerateOptions, Provider, ProviderConfig};
use gauss_core::rag;
use gauss_core::streaming::StreamEvent;
use gauss_core::telemetry;
use serde_json::json;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use wasm_bindgen::prelude::*;

thread_local! {
    static PROVIDERS: RefCell<HashMap<u32, Shared<dyn Provider>>> = RefCell::new(HashMap::new());
}

static NEXT_ID: AtomicU32 = AtomicU32::new(1);

/// Gauss Core version.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Creates a provider. Returns handle ID.
#[wasm_bindgen(js_name = "createProvider")]
pub fn create_provider(
    provider_type: &str,
    model: &str,
    api_key: &str,
    base_url: Option<String>,
) -> Result<u32, JsValue> {
    let mut config = ProviderConfig::new(api_key);
    if let Some(url) = base_url {
        config.base_url = Some(url);
    }

    let inner: Shared<dyn Provider> = match provider_type {
        "openai" => Shared::new(OpenAiProvider::new(model, config)),
        "anthropic" => Shared::new(AnthropicProvider::new(model, config)),
        "google" => Shared::new(GoogleProvider::new(model, config)),
        "groq" => Shared::new(GroqProvider::create(model, config)),
        "ollama" => Shared::new(OllamaProvider::create(model, config)),
        "deepseek" => Shared::new(DeepSeekProvider::create(model, config)),
        other => return Err(JsValue::from_str(&format!("Unknown provider: {other}"))),
    };

    let provider: Shared<dyn Provider> = Shared::new(RetryProvider::new(
        inner,
        RetryConfig {
            max_retries: 3,
            ..RetryConfig::default()
        },
    ));

    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    PROVIDERS.with(|p| p.borrow_mut().insert(id, provider));
    Ok(id)
}

/// Destroys a provider.
#[wasm_bindgen(js_name = "destroyProvider")]
pub fn destroy_provider(handle: u32) -> Result<(), JsValue> {
    PROVIDERS.with(|p| {
        p.borrow_mut()
            .remove(&handle)
            .ok_or_else(|| JsValue::from_str(&format!("Provider {handle} not found")))
    })?;
    Ok(())
}

fn get_provider(handle: u32) -> Result<Shared<dyn Provider>, JsValue> {
    PROVIDERS.with(|p| {
        p.borrow()
            .get(&handle)
            .cloned()
            .ok_or_else(|| JsValue::from_str(&format!("Provider {handle} not found")))
    })
}

fn parse_messages(json_str: &str) -> Result<Vec<RustMessage>, JsValue> {
    let js_msgs: Vec<serde_json::Value> = serde_json::from_str(json_str)
        .map_err(|e| JsValue::from_str(&format!("Invalid messages JSON: {e}")))?;
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

/// Call generate on a provider. Returns JSON string.
#[wasm_bindgen(js_name = "generate")]
pub async fn generate(
    provider_handle: u32,
    messages_json: &str,
    temperature: Option<f64>,
    max_tokens: Option<u32>,
) -> Result<String, JsValue> {
    let provider = get_provider(provider_handle)?;
    let rust_msgs = parse_messages(messages_json)?;

    let opts = GenerateOptions {
        temperature,
        max_tokens,
        ..GenerateOptions::default()
    };

    let result = provider
        .generate(&rust_msgs, &[], &opts)
        .await
        .map_err(|e| JsValue::from_str(&format!("Generate error: {e}")))?;

    let text = result.text().unwrap_or("").to_string();

    let output = json!({
        "text": text,
        "usage": {
            "inputTokens": result.usage.input_tokens,
            "outputTokens": result.usage.output_tokens,
        },
        "finishReason": format!("{:?}", result.finish_reason),
    });

    serde_json::to_string(&output).map_err(|e| JsValue::from_str(&format!("Serialize error: {e}")))
}

/// Stream generate on a provider. Returns a browser ReadableStream of JSON-serialized StreamEvents.
#[wasm_bindgen(js_name = "generateStream")]
pub async fn generate_stream(
    provider_handle: u32,
    messages_json: &str,
    temperature: Option<f64>,
    max_tokens: Option<u32>,
) -> Result<web_sys::ReadableStream, JsValue> {
    let provider = get_provider(provider_handle)?;
    let rust_msgs = parse_messages(messages_json)?;

    let opts = GenerateOptions {
        temperature,
        max_tokens,
        ..GenerateOptions::default()
    };

    let mut stream = provider
        .stream(&rust_msgs, &[], &opts)
        .await
        .map_err(|e| JsValue::from_str(&format!("Stream error: {e}")))?;

    let pull = Closure::wrap(Box::new(
        move |controller: web_sys::ReadableStreamDefaultController| -> js_sys::Promise {
            let controller = controller.clone();
            let stream_ref = unsafe {
                // SAFETY: WASM is single-threaded; the stream lives for the duration of the ReadableStream
                #[allow(clippy::deref_addrof)]
                &mut *(&raw mut stream)
            };
            wasm_bindgen_futures::future_to_promise(async move {
                match StreamExt::next(stream_ref).await {
                    Some(Ok(event)) => {
                        let json = serde_json::to_string(&event)
                            .map_err(|e| JsValue::from_str(&format!("{e}")))?;
                        controller.enqueue_with_chunk(&JsValue::from_str(&json))?;
                        if matches!(event, StreamEvent::Done) {
                            controller.close()?;
                        }
                        Ok(JsValue::UNDEFINED)
                    }
                    Some(Err(e)) => {
                        controller.error_with_e(&JsValue::from_str(&format!("{e}")));
                        Err(JsValue::from_str(&format!("{e}")))
                    }
                    None => {
                        controller.close()?;
                        Ok(JsValue::UNDEFINED)
                    }
                }
            })
        },
    )
        as Box<dyn FnMut(web_sys::ReadableStreamDefaultController) -> js_sys::Promise>);

    let underlying_source = js_sys::Object::new();
    js_sys::Reflect::set(
        &underlying_source,
        &"pull".into(),
        pull.as_ref().unchecked_ref(),
    )?;
    pull.forget();

    web_sys::ReadableStream::new_with_underlying_source(&underlying_source)
        .map_err(|e| JsValue::from_str(&format!("ReadableStream creation error: {e:?}")))
}

/// Run an agent. Returns JSON string.
#[wasm_bindgen(js_name = "agentRun")]
pub async fn agent_run(
    name: &str,
    provider_handle: u32,
    messages_json: &str,
    options_json: Option<String>,
) -> Result<String, JsValue> {
    let provider = get_provider(provider_handle)?;
    let rust_msgs = parse_messages(messages_json)?;
    let mut builder = RustAgent::builder(name, provider);

    if let Some(opts_str) = options_json {
        let opts: serde_json::Value = serde_json::from_str(&opts_str)
            .map_err(|e| JsValue::from_str(&format!("Invalid options JSON: {e}")))?;

        if let Some(instructions) = opts["instructions"].as_str() {
            builder = builder.instructions(instructions);
        }
        if let Some(max_steps) = opts["maxSteps"].as_u64() {
            builder = builder.max_steps(max_steps as usize);
        }
        if let Some(temp) = opts["temperature"].as_f64() {
            builder = builder.temperature(temp);
        }
        if let Some(tp) = opts["topP"].as_f64() {
            builder = builder.top_p(tp);
        }
        if let Some(mt) = opts["maxTokens"].as_u64() {
            builder = builder.max_tokens(mt as u32);
        }
        if let Some(stop_tool) = opts["stopOnTool"].as_str() {
            builder = builder.stop_when(StopCondition::HasToolCall(stop_tool.to_string()));
        }
        if let Some(schema) = opts.get("outputSchema").filter(|s| !s.is_null()) {
            builder = builder.output_schema(schema.clone());
        }
    }

    let agent = builder.build();
    let output = agent
        .run(rust_msgs)
        .await
        .map_err(|e| JsValue::from_str(&format!("Agent error: {e}")))?;

    let result = json!({
        "text": output.text,
        "steps": output.steps,
        "usage": {
            "inputTokens": output.usage.input_tokens,
            "outputTokens": output.usage.output_tokens,
        },
        "structuredOutput": output.structured_output,
    });

    serde_json::to_string(&result).map_err(|e| JsValue::from_str(&format!("Serialize error: {e}")))
}

// ============ WASM Registries ============

thread_local! {
    static MEMORIES: RefCell<HashMap<u32, Shared<memory::InMemoryMemory>>> = RefCell::new(HashMap::new());
    static VECTOR_STORES: RefCell<HashMap<u32, Shared<rag::InMemoryVectorStore>>> = RefCell::new(HashMap::new());
    static MCP_SERVERS: RefCell<HashMap<u32, Shared<RefCell<mcp::McpServer>>>> = RefCell::new(HashMap::new());
    static APPROVALS: RefCell<HashMap<u32, Shared<RefCell<hitl::ApprovalManager>>>> = RefCell::new(HashMap::new());
    static CHECKPOINTS: RefCell<HashMap<u32, Shared<hitl::InMemoryCheckpointStore>>> = RefCell::new(HashMap::new());
    static EVALS: RefCell<HashMap<u32, Shared<RefCell<eval::EvalRunner>>>> = RefCell::new(HashMap::new());
    static COLLECTORS: RefCell<HashMap<u32, Shared<RefCell<telemetry::TelemetryCollector>>>> = RefCell::new(HashMap::new());
}

fn next_handle() -> u32 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

fn err(msg: &str) -> JsValue {
    JsValue::from_str(msg)
}

// ============ Memory ============

#[wasm_bindgen(js_name = "createMemory")]
pub fn create_memory() -> u32 {
    let id = next_handle();
    MEMORIES.with(|m| {
        m.borrow_mut()
            .insert(id, Shared::new(memory::InMemoryMemory::new()))
    });
    id
}

#[wasm_bindgen(js_name = "memoryStore")]
pub async fn memory_store(handle: u32, entry_json: String) -> Result<(), JsValue> {
    use memory::Memory;
    let mem = MEMORIES
        .with(|m| m.borrow().get(&handle).cloned())
        .ok_or_else(|| err("Memory not found"))?;
    let entry: memory::MemoryEntry =
        serde_json::from_str(&entry_json).map_err(|e| err(&format!("Invalid entry: {e}")))?;
    mem.store(entry).await.map_err(|e| err(&format!("{e}")))
}

#[wasm_bindgen(js_name = "memoryRecall")]
pub async fn memory_recall(handle: u32, options_json: Option<String>) -> Result<String, JsValue> {
    use memory::Memory;
    let mem = MEMORIES
        .with(|m| m.borrow().get(&handle).cloned())
        .ok_or_else(|| err("Memory not found"))?;
    let opts: memory::RecallOptions = match options_json {
        Some(j) => serde_json::from_str(&j).map_err(|e| err(&format!("Invalid options: {e}")))?,
        None => memory::RecallOptions::default(),
    };
    let entries = mem.recall(opts).await.map_err(|e| err(&format!("{e}")))?;
    serde_json::to_string(&entries).map_err(|e| err(&format!("{e}")))
}

#[wasm_bindgen(js_name = "memoryClear")]
pub async fn memory_clear(handle: u32, session_id: Option<String>) -> Result<(), JsValue> {
    let mem = MEMORIES
        .with(|m| m.borrow().get(&handle).cloned())
        .ok_or_else(|| err("Memory not found"))?;
    memory::Memory::clear(&*mem, session_id.as_deref())
        .await
        .map_err(|e| err(&format!("{e}")))
}

#[wasm_bindgen(js_name = "destroyMemory")]
pub fn destroy_memory(handle: u32) -> Result<(), JsValue> {
    MEMORIES
        .with(|m| m.borrow_mut().remove(&handle))
        .ok_or_else(|| err("Memory not found"))?;
    Ok(())
}

// ============ Context ============

#[wasm_bindgen(js_name = "countTokens")]
pub fn count_tokens(text: &str) -> u32 {
    context::count_tokens(text) as u32
}

#[wasm_bindgen(js_name = "countMessageTokens")]
pub fn count_message_tokens(messages_json: &str) -> Result<u32, JsValue> {
    let msgs = parse_messages(messages_json)?;
    Ok(context::count_messages_tokens(&msgs) as u32)
}

#[wasm_bindgen(js_name = "getContextWindowSize")]
pub fn get_context_window_size(model: &str) -> u32 {
    context::context_window_size(model) as u32
}

// ============ RAG ============

#[wasm_bindgen(js_name = "createVectorStore")]
pub fn create_vector_store() -> u32 {
    let id = next_handle();
    VECTOR_STORES.with(|v| {
        v.borrow_mut()
            .insert(id, Shared::new(rag::InMemoryVectorStore::new()))
    });
    id
}

#[wasm_bindgen(js_name = "vectorStoreUpsert")]
pub async fn vector_store_upsert(handle: u32, chunks_json: String) -> Result<(), JsValue> {
    use rag::VectorStore;
    let store = VECTOR_STORES
        .with(|v| v.borrow().get(&handle).cloned())
        .ok_or_else(|| err("VectorStore not found"))?;
    let chunks: Vec<rag::Chunk> =
        serde_json::from_str(&chunks_json).map_err(|e| err(&format!("Invalid chunks: {e}")))?;
    store.upsert(chunks).await.map_err(|e| err(&format!("{e}")))
}

#[wasm_bindgen(js_name = "vectorStoreSearch")]
pub async fn vector_store_search(
    handle: u32,
    embedding_json: String,
    top_k: u32,
) -> Result<String, JsValue> {
    use rag::VectorStore;
    let store = VECTOR_STORES
        .with(|v| v.borrow().get(&handle).cloned())
        .ok_or_else(|| err("VectorStore not found"))?;
    let embedding: Vec<f32> = serde_json::from_str(&embedding_json)
        .map_err(|e| err(&format!("Invalid embedding: {e}")))?;
    let results = store
        .search(&embedding, top_k as usize)
        .await
        .map_err(|e| err(&format!("{e}")))?;
    serde_json::to_string(&results).map_err(|e| err(&format!("{e}")))
}

#[wasm_bindgen(js_name = "destroyVectorStore")]
pub fn destroy_vector_store(handle: u32) -> Result<(), JsValue> {
    VECTOR_STORES
        .with(|v| v.borrow_mut().remove(&handle))
        .ok_or_else(|| err("VectorStore not found"))?;
    Ok(())
}

#[wasm_bindgen(js_name = "cosineSimilarity")]
pub fn cosine_similarity(a_json: &str, b_json: &str) -> Result<f64, JsValue> {
    let a: Vec<f32> = serde_json::from_str(a_json).map_err(|e| err(&format!("{e}")))?;
    let b: Vec<f32> = serde_json::from_str(b_json).map_err(|e| err(&format!("{e}")))?;
    Ok(rag::cosine_similarity(&a, &b) as f64)
}

// ============ MCP ============

#[wasm_bindgen(js_name = "createMcpServer")]
pub fn create_mcp_server(name: &str, version_str: &str) -> u32 {
    let id = next_handle();
    MCP_SERVERS.with(|s| {
        s.borrow_mut().insert(
            id,
            Shared::new(RefCell::new(mcp::McpServer::new(name, version_str))),
        )
    });
    id
}

#[wasm_bindgen(js_name = "mcpServerAddTool")]
pub fn mcp_server_add_tool(handle: u32, tool_json: String) -> Result<(), JsValue> {
    let server = MCP_SERVERS
        .with(|s| s.borrow().get(&handle).cloned())
        .ok_or_else(|| err("McpServer not found"))?;
    let mcp_tool: mcp::McpTool =
        serde_json::from_str(&tool_json).map_err(|e| err(&format!("Invalid tool: {e}")))?;
    let gauss_tool = mcp::mcp_tool_to_gauss(&mcp_tool);
    server.borrow_mut().add_tool(gauss_tool);
    Ok(())
}

#[wasm_bindgen(js_name = "mcpServerHandle")]
pub async fn mcp_server_handle(handle: u32, message_json: String) -> Result<String, JsValue> {
    let server = MCP_SERVERS
        .with(|s| s.borrow().get(&handle).cloned())
        .ok_or_else(|| err("McpServer not found"))?;
    let msg: mcp::JsonRpcMessage =
        serde_json::from_str(&message_json).map_err(|e| err(&format!("Invalid message: {e}")))?;
    let resp = server
        .borrow()
        .handle_message(msg)
        .await
        .map_err(|e| err(&format!("{e}")))?;
    serde_json::to_string(&resp).map_err(|e| err(&format!("{e}")))
}

#[wasm_bindgen(js_name = "destroyMcpServer")]
pub fn destroy_mcp_server(handle: u32) -> Result<(), JsValue> {
    MCP_SERVERS
        .with(|s| s.borrow_mut().remove(&handle))
        .ok_or_else(|| err("McpServer not found"))?;
    Ok(())
}

// ============ HITL ============

#[wasm_bindgen(js_name = "createApprovalManager")]
pub fn create_approval_manager() -> u32 {
    let id = next_handle();
    APPROVALS.with(|a| {
        a.borrow_mut()
            .insert(id, Shared::new(RefCell::new(hitl::ApprovalManager::new())))
    });
    id
}

#[wasm_bindgen(js_name = "approvalRequest")]
pub fn approval_request(
    handle: u32,
    tool_name: String,
    args_json: String,
    session_id: String,
) -> Result<String, JsValue> {
    let mgr = APPROVALS
        .with(|a| a.borrow().get(&handle).cloned())
        .ok_or_else(|| err("ApprovalManager not found"))?;
    let args: serde_json::Value =
        serde_json::from_str(&args_json).map_err(|e| err(&format!("Invalid args: {e}")))?;
    let req = mgr
        .borrow()
        .request_approval(tool_name, args, 0, session_id)
        .map_err(|e| err(&format!("{e}")))?;
    Ok(req.id.clone())
}

#[wasm_bindgen(js_name = "approvalApprove")]
pub fn approval_approve(
    handle: u32,
    request_id: &str,
    modified_args: Option<String>,
) -> Result<(), JsValue> {
    let mgr = APPROVALS
        .with(|a| a.borrow().get(&handle).cloned())
        .ok_or_else(|| err("ApprovalManager not found"))?;
    let args: Option<serde_json::Value> = match modified_args {
        Some(j) => Some(serde_json::from_str(&j).map_err(|e| err(&format!("{e}")))?),
        None => None,
    };
    mgr.borrow()
        .approve(request_id, args)
        .map_err(|e| err(&format!("{e}")))?;
    Ok(())
}

#[wasm_bindgen(js_name = "approvalDeny")]
pub fn approval_deny(handle: u32, request_id: &str, reason: Option<String>) -> Result<(), JsValue> {
    let mgr = APPROVALS
        .with(|a| a.borrow().get(&handle).cloned())
        .ok_or_else(|| err("ApprovalManager not found"))?;
    mgr.borrow()
        .deny(request_id, reason)
        .map_err(|e| err(&format!("{e}")))?;
    Ok(())
}

#[wasm_bindgen(js_name = "approvalListPending")]
pub fn approval_list_pending(handle: u32) -> Result<String, JsValue> {
    let mgr = APPROVALS
        .with(|a| a.borrow().get(&handle).cloned())
        .ok_or_else(|| err("ApprovalManager not found"))?;
    let pending = mgr
        .borrow()
        .list_pending()
        .map_err(|e| err(&format!("{e}")))?;
    serde_json::to_string(&pending).map_err(|e| err(&format!("{e}")))
}

#[wasm_bindgen(js_name = "destroyApprovalManager")]
pub fn destroy_approval_manager(handle: u32) -> Result<(), JsValue> {
    APPROVALS
        .with(|a| a.borrow_mut().remove(&handle))
        .ok_or_else(|| err("ApprovalManager not found"))?;
    Ok(())
}

// ============ Checkpoint Store ============

#[wasm_bindgen(js_name = "createCheckpointStore")]
pub fn create_checkpoint_store() -> u32 {
    let id = next_handle();
    CHECKPOINTS.with(|c| {
        c.borrow_mut()
            .insert(id, Shared::new(hitl::InMemoryCheckpointStore::new()))
    });
    id
}

#[wasm_bindgen(js_name = "checkpointSave")]
pub async fn checkpoint_save(handle: u32, checkpoint_json: String) -> Result<(), JsValue> {
    use hitl::CheckpointStore;
    let store = CHECKPOINTS
        .with(|c| c.borrow().get(&handle).cloned())
        .ok_or_else(|| err("CheckpointStore not found"))?;
    let cp: hitl::Checkpoint = serde_json::from_str(&checkpoint_json)
        .map_err(|e| err(&format!("Invalid checkpoint: {e}")))?;
    store.save(&cp).await.map_err(|e| err(&format!("{e}")))
}

#[wasm_bindgen(js_name = "checkpointLoad")]
pub async fn checkpoint_load(handle: u32, checkpoint_id: &str) -> Result<String, JsValue> {
    use hitl::CheckpointStore;
    let store = CHECKPOINTS
        .with(|c| c.borrow().get(&handle).cloned())
        .ok_or_else(|| err("CheckpointStore not found"))?;
    let cp = store
        .load(checkpoint_id)
        .await
        .map_err(|e| err(&format!("{e}")))?;
    serde_json::to_string(&cp).map_err(|e| err(&format!("{e}")))
}

#[wasm_bindgen(js_name = "destroyCheckpointStore")]
pub fn destroy_checkpoint_store(handle: u32) -> Result<(), JsValue> {
    CHECKPOINTS
        .with(|c| c.borrow_mut().remove(&handle))
        .ok_or_else(|| err("CheckpointStore not found"))?;
    Ok(())
}

// ============ Eval ============

#[wasm_bindgen(js_name = "createEvalRunner")]
pub fn create_eval_runner(threshold: Option<f64>) -> u32 {
    let id = next_handle();
    let mut runner = eval::EvalRunner::new();
    if let Some(t) = threshold {
        runner = runner.with_threshold(t);
    }
    EVALS.with(|e| e.borrow_mut().insert(id, Shared::new(RefCell::new(runner))));
    id
}

#[wasm_bindgen(js_name = "evalAddScorer")]
pub fn eval_add_scorer(handle: u32, scorer_type: &str) -> Result<(), JsValue> {
    let runner = EVALS
        .with(|e| e.borrow().get(&handle).cloned())
        .ok_or_else(|| err("EvalRunner not found"))?;
    let scorer: Shared<dyn eval::Scorer> = match scorer_type {
        "exact_match" => Shared::new(eval::ExactMatchScorer),
        "contains" => Shared::new(eval::ContainsScorer),
        "length_ratio" => Shared::new(eval::LengthRatioScorer),
        other => return Err(err(&format!("Unknown scorer: {other}"))),
    };
    runner.borrow_mut().add_scorer(scorer);
    Ok(())
}

#[wasm_bindgen(js_name = "loadDatasetJsonl")]
pub fn load_dataset_jsonl(jsonl: &str) -> Result<String, JsValue> {
    let cases = eval::load_dataset_jsonl(jsonl).map_err(|e| err(&format!("{e}")))?;
    serde_json::to_string(&cases).map_err(|e| err(&format!("{e}")))
}

#[wasm_bindgen(js_name = "loadDatasetJson")]
pub fn load_dataset_json(json_str: &str) -> Result<String, JsValue> {
    let cases = eval::load_dataset_json(json_str).map_err(|e| err(&format!("{e}")))?;
    serde_json::to_string(&cases).map_err(|e| err(&format!("{e}")))
}

#[wasm_bindgen(js_name = "destroyEvalRunner")]
pub fn destroy_eval_runner(handle: u32) -> Result<(), JsValue> {
    EVALS
        .with(|e| e.borrow_mut().remove(&handle))
        .ok_or_else(|| err("EvalRunner not found"))?;
    Ok(())
}

// ============ Telemetry ============

#[wasm_bindgen(js_name = "createTelemetry")]
pub fn create_telemetry() -> u32 {
    let id = next_handle();
    COLLECTORS.with(|c| {
        c.borrow_mut().insert(
            id,
            Shared::new(RefCell::new(telemetry::TelemetryCollector::new())),
        )
    });
    id
}

#[wasm_bindgen(js_name = "telemetryRecordSpan")]
pub fn telemetry_record_span(handle: u32, span_json: &str) -> Result<(), JsValue> {
    let coll = COLLECTORS
        .with(|c| c.borrow().get(&handle).cloned())
        .ok_or_else(|| err("TelemetryCollector not found"))?;
    let span: telemetry::SpanRecord =
        serde_json::from_str(span_json).map_err(|e| err(&format!("Invalid span: {e}")))?;
    coll.borrow().record_span(span);
    Ok(())
}

#[wasm_bindgen(js_name = "telemetryExportSpans")]
pub fn telemetry_export_spans(handle: u32) -> Result<String, JsValue> {
    let coll = COLLECTORS
        .with(|c| c.borrow().get(&handle).cloned())
        .ok_or_else(|| err("TelemetryCollector not found"))?;
    let spans = coll.borrow().export_spans();
    serde_json::to_string(&spans).map_err(|e| err(&format!("{e}")))
}

#[wasm_bindgen(js_name = "telemetryExportMetrics")]
pub fn telemetry_export_metrics(handle: u32) -> Result<String, JsValue> {
    let coll = COLLECTORS
        .with(|c| c.borrow().get(&handle).cloned())
        .ok_or_else(|| err("TelemetryCollector not found"))?;
    let metrics = coll.borrow().export_metrics();
    serde_json::to_string(&metrics).map_err(|e| err(&format!("{e}")))
}

#[wasm_bindgen(js_name = "telemetryClear")]
pub fn telemetry_clear(handle: u32) -> Result<(), JsValue> {
    let coll = COLLECTORS
        .with(|c| c.borrow().get(&handle).cloned())
        .ok_or_else(|| err("TelemetryCollector not found"))?;
    coll.borrow().clear();
    Ok(())
}

#[wasm_bindgen(js_name = "destroyTelemetry")]
pub fn destroy_telemetry(handle: u32) -> Result<(), JsValue> {
    COLLECTORS
        .with(|c| c.borrow_mut().remove(&handle))
        .ok_or_else(|| err("TelemetryCollector not found"))?;
    Ok(())
}

// ============ Stream Transform (WASM) ============

use gauss_core::stream_transform;

#[wasm_bindgen(js_name = "parsePartialJson")]
pub fn parse_partial_json(text: &str) -> Option<String> {
    stream_transform::parse_partial_json(text).map(|v| v.to_string())
}

// ============ Plugin System (WASM) ============

use gauss_core::plugin;

thread_local! {
    static PLUGIN_REGISTRIES: RefCell<HashMap<u32, plugin::PluginRegistry>> = RefCell::new(HashMap::new());
}

#[wasm_bindgen(js_name = "createPluginRegistry")]
pub fn create_plugin_registry() -> u32 {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    PLUGIN_REGISTRIES.with(|r| {
        r.borrow_mut().insert(id, plugin::PluginRegistry::new());
    });
    id
}

#[wasm_bindgen(js_name = "pluginRegistryAddTelemetry")]
pub fn plugin_registry_add_telemetry(handle: u32) -> Result<(), JsValue> {
    PLUGIN_REGISTRIES.with(|r| {
        let mut reg = r.borrow_mut();
        let registry = reg
            .get_mut(&handle)
            .ok_or_else(|| err("PluginRegistry not found"))?;
        registry.register(Shared::new(plugin::TelemetryPlugin));
        Ok(())
    })
}

#[wasm_bindgen(js_name = "pluginRegistryAddMemory")]
pub fn plugin_registry_add_memory(handle: u32) -> Result<(), JsValue> {
    PLUGIN_REGISTRIES.with(|r| {
        let mut reg = r.borrow_mut();
        let registry = reg
            .get_mut(&handle)
            .ok_or_else(|| err("PluginRegistry not found"))?;
        registry.register(Shared::new(plugin::MemoryPlugin));
        Ok(())
    })
}

#[wasm_bindgen(js_name = "pluginRegistryList")]
pub fn plugin_registry_list(handle: u32) -> Result<String, JsValue> {
    PLUGIN_REGISTRIES.with(|r| {
        let reg = r.borrow();
        let registry = reg
            .get(&handle)
            .ok_or_else(|| err("PluginRegistry not found"))?;
        let names: Vec<&str> = registry.list();
        Ok(serde_json::to_string(&names).unwrap_or_default())
    })
}

#[wasm_bindgen(js_name = "destroyPluginRegistry")]
pub fn destroy_plugin_registry(handle: u32) -> Result<(), JsValue> {
    PLUGIN_REGISTRIES
        .with(|r| r.borrow_mut().remove(&handle))
        .ok_or_else(|| err("PluginRegistry not found"))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tool Validator (patterns module)
// ---------------------------------------------------------------------------

use gauss_core::patterns::{CoercionStrategy, ToolValidator as RustToolValidator};

thread_local! {
    static TOOL_VALIDATORS: RefCell<HashMap<u32, RustToolValidator>> = RefCell::new(HashMap::new());
}

#[wasm_bindgen(js_name = "createToolValidator")]
pub fn create_tool_validator(strategies: Option<Vec<String>>) -> u32 {
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
    TOOL_VALIDATORS.with(|r| r.borrow_mut().insert(id, validator));
    id
}

#[wasm_bindgen(js_name = "toolValidatorValidate")]
pub fn tool_validator_validate(
    handle: u32,
    input: String,
    schema: String,
) -> Result<String, JsValue> {
    let input_val: serde_json::Value =
        serde_json::from_str(&input).map_err(|e| err(&format!("{e}")))?;
    let schema_val: serde_json::Value =
        serde_json::from_str(&schema).map_err(|e| err(&format!("{e}")))?;
    TOOL_VALIDATORS.with(|r| {
        let reg = r.borrow();
        let validator = reg
            .get(&handle)
            .ok_or_else(|| err("ToolValidator not found"))?;
        let result = validator
            .validate(input_val, &schema_val)
            .map_err(|e| err(&format!("{e}")))?;
        serde_json::to_string(&result).map_err(|e| err(&format!("{e}")))
    })
}

#[wasm_bindgen(js_name = "destroyToolValidator")]
pub fn destroy_tool_validator(handle: u32) -> Result<(), JsValue> {
    TOOL_VALIDATORS
        .with(|r| r.borrow_mut().remove(&handle))
        .ok_or_else(|| err("ToolValidator not found"))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Agent Config (config module)
// ---------------------------------------------------------------------------

#[wasm_bindgen(js_name = "agentConfigFromJson")]
pub fn agent_config_from_json(json_str: String) -> Result<String, JsValue> {
    let config =
        gauss_core::config::AgentConfig::from_json(&json_str).map_err(|e| err(&format!("{e}")))?;
    config.to_json().map_err(|e| err(&format!("{e}")))
}

#[wasm_bindgen(js_name = "agentConfigResolveEnv")]
pub fn agent_config_resolve_env(value: String) -> String {
    gauss_core::config::resolve_env(&value)
}
