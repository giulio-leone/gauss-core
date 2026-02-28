#[macro_use]
extern crate napi_derive;

use gauss_core::agent::{Agent as RustAgent, AgentOutput as RustAgentOutput, StopCondition};
use gauss_core::context;
use gauss_core::eval;
use gauss_core::hitl;
use gauss_core::mcp;
use gauss_core::memory;
use gauss_core::message::Message as RustMessage;
use gauss_core::middleware;
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
use gauss_core::tool::Tool as RustTool;
use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ErrorStrategy, ThreadsafeFunction};
use serde_json::json;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

/// Global provider registry — maps handle IDs to provider instances.
static NEXT_ID: AtomicU32 = AtomicU32::new(1);

fn providers() -> &'static Mutex<HashMap<u32, Arc<dyn Provider>>> {
    use std::sync::OnceLock;
    static PROVIDERS: OnceLock<Mutex<HashMap<u32, Arc<dyn Provider>>>> = OnceLock::new();
    PROVIDERS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Gauss Core version.
#[napi]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// ============ Provider ============

#[napi(object)]
pub struct ProviderOptions {
    pub api_key: String,
    pub base_url: Option<String>,
    pub timeout_ms: Option<u32>,
    pub max_retries: Option<u32>,
    pub organization: Option<String>,
}

/// Creates a provider and returns its handle ID.
/// Supported: "openai", "anthropic", "google", "groq", "ollama", "deepseek"
#[napi]
pub fn create_provider(
    provider_type: String,
    model: String,
    options: ProviderOptions,
) -> Result<u32> {
    let mut config = ProviderConfig::new(&options.api_key);
    if let Some(url) = options.base_url {
        config.base_url = Some(url);
    }
    if let Some(timeout) = options.timeout_ms {
        config.timeout_ms = Some(timeout as u64);
    }
    if let Some(org) = options.organization {
        config.organization = Some(org);
    }
    config.max_retries = options.max_retries;

    let max_retries = config.max_retries.unwrap_or(3);

    let inner: Arc<dyn Provider> = match provider_type.as_str() {
        "openai" => Arc::new(OpenAiProvider::new(model, config)),
        "anthropic" => Arc::new(AnthropicProvider::new(model, config)),
        "google" => Arc::new(GoogleProvider::new(model, config)),
        "groq" => Arc::new(GroqProvider::create(model, config)),
        "ollama" => Arc::new(OllamaProvider::create(model, config)),
        "deepseek" => Arc::new(DeepSeekProvider::create(model, config)),
        other => {
            return Err(napi::Error::from_reason(format!(
                "Unknown provider type: {other}"
            )));
        }
    };

    let provider: Arc<dyn Provider> = if max_retries > 0 {
        Arc::new(RetryProvider::new(
            inner,
            RetryConfig {
                max_retries,
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

/// Destroys a provider and frees its resources.
#[napi]
pub fn destroy_provider(handle: u32) -> Result<()> {
    providers()
        .lock()
        .unwrap()
        .remove(&handle)
        .ok_or_else(|| napi::Error::from_reason(format!("Provider {handle} not found")))?;
    Ok(())
}

fn get_provider(handle: u32) -> Result<Arc<dyn Provider>> {
    providers()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| napi::Error::from_reason(format!("Provider {handle} not found")))
}

// ============ Tool ============

#[napi(object)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: Option<serde_json::Value>,
}

// ============ Message ============

#[napi(object)]
pub struct JsMessage {
    pub role: String,
    pub content: String,
}

fn js_message_to_rust(msg: &JsMessage) -> RustMessage {
    match msg.role.as_str() {
        "system" => RustMessage::system(&msg.content),
        "assistant" => RustMessage::assistant(&msg.content),
        "user" => RustMessage::user(&msg.content),
        _ => RustMessage::user(&msg.content),
    }
}

// ============ Agent Options ============

#[napi(object)]
pub struct AgentOptions {
    pub instructions: Option<String>,
    pub max_steps: Option<u32>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_tokens: Option<u32>,
    pub seed: Option<f64>,
    pub stop_on_tool: Option<String>,
    pub output_schema: Option<serde_json::Value>,
}

// ============ Agent Output ============

#[napi(object)]
pub struct AgentResult {
    pub text: String,
    pub steps: u32,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub structured_output: Option<serde_json::Value>,
}

fn rust_output_to_js(output: RustAgentOutput) -> AgentResult {
    AgentResult {
        text: output.text,
        steps: output.steps as u32,
        input_tokens: output.usage.input_tokens as u32,
        output_tokens: output.usage.output_tokens as u32,
        structured_output: output.structured_output,
    }
}

// ============ Agent.run() ============

/// Run an agent with the given provider, tools, messages, and options.
#[napi]
pub async fn agent_run(
    name: String,
    provider_handle: u32,
    tools: Vec<ToolDef>,
    messages: Vec<JsMessage>,
    options: Option<AgentOptions>,
) -> Result<AgentResult> {
    let provider = get_provider(provider_handle)?;
    let opts = options.unwrap_or(AgentOptions {
        instructions: None,
        max_steps: None,
        temperature: None,
        top_p: None,
        max_tokens: None,
        seed: None,
        stop_on_tool: None,
        output_schema: None,
    });

    let mut builder = RustAgent::builder(name, provider);

    if let Some(instructions) = opts.instructions {
        builder = builder.instructions(instructions);
    }
    if let Some(max_steps) = opts.max_steps {
        builder = builder.max_steps(max_steps as usize);
    }
    if let Some(temp) = opts.temperature {
        builder = builder.temperature(temp);
    }
    if let Some(tp) = opts.top_p {
        builder = builder.top_p(tp);
    }
    if let Some(mt) = opts.max_tokens {
        builder = builder.max_tokens(mt);
    }
    if let Some(seed) = opts.seed {
        builder = builder.seed(seed as u64);
    }
    if let Some(ref schema) = opts.output_schema {
        builder = builder.output_schema(schema.clone());
    }
    if let Some(tool_name) = opts.stop_on_tool {
        builder = builder.stop_when(StopCondition::HasToolCall(tool_name));
    }

    for td in &tools {
        let mut tool_builder = RustTool::builder(&td.name, &td.description);
        if let Some(ref params) = td.parameters {
            tool_builder = tool_builder.parameters_json(params.clone());
        }
        builder = builder.tool(tool_builder.build());
    }

    let agent = builder.build();
    let rust_messages: Vec<RustMessage> = messages.iter().map(js_message_to_rust).collect();

    let output = agent
        .run(rust_messages)
        .await
        .map_err(|e| napi::Error::from_reason(format!("Agent error: {e}")))?;

    Ok(rust_output_to_js(output))
}

// ============ Agent.run() with JS Tool Execution ============

/// Run an agent where tool execution is delegated back to JavaScript.
/// The `tool_executor` callback is invoked for each tool call:
///   (callJson: string) => Promise<string>
/// callJson = `{"tool":"<name>","args":{...}}`
/// Return value = JSON string of the tool result.
#[napi]
pub async fn agent_run_with_tool_executor(
    name: String,
    provider_handle: u32,
    tools: Vec<ToolDef>,
    messages: Vec<JsMessage>,
    options: Option<AgentOptions>,
    #[napi(ts_arg_type = "(callJson: string) => Promise<string>")]
    tool_executor: ThreadsafeFunction<String, ErrorStrategy::Fatal>,
) -> Result<AgentResult> {
    let provider = get_provider(provider_handle)?;
    let opts = options.unwrap_or(AgentOptions {
        instructions: None,
        max_steps: None,
        temperature: None,
        top_p: None,
        max_tokens: None,
        seed: None,
        stop_on_tool: None,
        output_schema: None,
    });

    let mut builder = RustAgent::builder(name, provider);

    if let Some(instructions) = opts.instructions {
        builder = builder.instructions(instructions);
    }
    if let Some(max_steps) = opts.max_steps {
        builder = builder.max_steps(max_steps as usize);
    }
    if let Some(temp) = opts.temperature {
        builder = builder.temperature(temp);
    }
    if let Some(tp) = opts.top_p {
        builder = builder.top_p(tp);
    }
    if let Some(mt) = opts.max_tokens {
        builder = builder.max_tokens(mt);
    }
    if let Some(seed) = opts.seed {
        builder = builder.seed(seed as u64);
    }
    if let Some(ref schema) = opts.output_schema {
        builder = builder.output_schema(schema.clone());
    }
    if let Some(tool_name) = opts.stop_on_tool {
        builder = builder.stop_when(StopCondition::HasToolCall(tool_name));
    }

    let tool_executor = Arc::new(tool_executor);

    for td in &tools {
        let tool_exec = tool_executor.clone();
        let tool_name = td.name.clone();

        let mut tool_builder = RustTool::builder(&td.name, &td.description);
        if let Some(ref params) = td.parameters {
            tool_builder = tool_builder.parameters_json(params.clone());
        }

        tool_builder = tool_builder.execute(move |args: serde_json::Value| {
            let exec = tool_exec.clone();
            let tn = tool_name.clone();
            async move {
                let call_json = serde_json::to_string(&json!({
                    "tool": tn,
                    "args": args
                }))
                .map_err(|e| gauss_core::error::GaussError::tool(&tn, format!("Serialize: {e}")))?;

                let promise: Promise<String> = exec
                    .call_async::<Promise<String>>(call_json)
                    .await
                    .map_err(|e| {
                        gauss_core::error::GaussError::tool(&tn, format!("NAPI call error: {e}"))
                    })?;

                let result_str = promise.await.map_err(|e| {
                    gauss_core::error::GaussError::tool(&tn, format!("JS Promise error: {e}"))
                })?;

                serde_json::from_str(&result_str).map_err(|e| {
                    gauss_core::error::GaussError::tool(&tn, format!("Deserialize: {e}"))
                })
            }
        });

        builder = builder.tool(tool_builder.build());
    }

    let agent = builder.build();
    let rust_messages: Vec<RustMessage> = messages.iter().map(js_message_to_rust).collect();

    let output = agent
        .run(rust_messages)
        .await
        .map_err(|e| napi::Error::from_reason(format!("Agent error: {e}")))?;

    Ok(rust_output_to_js(output))
}

// ============ Agent.stream() with JS Tool Execution ============

/// Stream an agent execution, pushing events to JS via callback.
/// Each event is a JSON string: `{"type":"text_delta","step":0,"delta":"..."}`
/// The stream_callback receives events as they happen.
/// Returns the final AgentResult when the stream completes.
#[napi]
pub async fn agent_stream_with_tool_executor(
    name: String,
    provider_handle: u32,
    tools: Vec<ToolDef>,
    messages: Vec<JsMessage>,
    options: Option<AgentOptions>,
    #[napi(ts_arg_type = "(eventJson: string) => void")] stream_callback: ThreadsafeFunction<
        String,
        ErrorStrategy::Fatal,
    >,
    #[napi(ts_arg_type = "(callJson: string) => Promise<string>")]
    tool_executor: ThreadsafeFunction<String, ErrorStrategy::Fatal>,
) -> Result<AgentResult> {
    use futures::StreamExt;

    let provider = get_provider(provider_handle)?;
    let opts = options.unwrap_or(AgentOptions {
        instructions: None,
        max_steps: None,
        temperature: None,
        top_p: None,
        max_tokens: None,
        seed: None,
        stop_on_tool: None,
        output_schema: None,
    });

    let mut builder = RustAgent::builder(name, provider);

    if let Some(instructions) = opts.instructions {
        builder = builder.instructions(instructions);
    }
    if let Some(max_steps) = opts.max_steps {
        builder = builder.max_steps(max_steps as usize);
    }
    if let Some(temp) = opts.temperature {
        builder = builder.temperature(temp);
    }
    if let Some(tp) = opts.top_p {
        builder = builder.top_p(tp);
    }
    if let Some(mt) = opts.max_tokens {
        builder = builder.max_tokens(mt);
    }
    if let Some(seed) = opts.seed {
        builder = builder.seed(seed as u64);
    }
    if let Some(ref schema) = opts.output_schema {
        builder = builder.output_schema(schema.clone());
    }
    if let Some(tool_name) = opts.stop_on_tool {
        builder = builder.stop_when(StopCondition::HasToolCall(tool_name));
    }

    let tool_executor = Arc::new(tool_executor);

    for td in &tools {
        let tool_exec = tool_executor.clone();
        let tool_name = td.name.clone();

        let mut tool_builder = RustTool::builder(&td.name, &td.description);
        if let Some(ref params) = td.parameters {
            tool_builder = tool_builder.parameters_json(params.clone());
        }

        tool_builder = tool_builder.execute(move |args: serde_json::Value| {
            let exec = tool_exec.clone();
            let tn = tool_name.clone();
            async move {
                let call_json = serde_json::to_string(&json!({
                    "tool": tn,
                    "args": args
                }))
                .map_err(|e| gauss_core::error::GaussError::tool(&tn, format!("Serialize: {e}")))?;

                let promise: Promise<String> = exec
                    .call_async::<Promise<String>>(call_json)
                    .await
                    .map_err(|e| {
                        gauss_core::error::GaussError::tool(&tn, format!("NAPI call error: {e}"))
                    })?;

                let result_str = promise.await.map_err(|e| {
                    gauss_core::error::GaussError::tool(&tn, format!("JS Promise error: {e}"))
                })?;

                serde_json::from_str(&result_str).map_err(|e| {
                    gauss_core::error::GaussError::tool(&tn, format!("Deserialize: {e}"))
                })
            }
        });

        builder = builder.tool(tool_builder.build());
    }

    let agent = builder.build();
    let rust_messages: Vec<RustMessage> = messages.iter().map(js_message_to_rust).collect();

    let mut stream = agent
        .run_stream(rust_messages)
        .await
        .map_err(|e| napi::Error::from_reason(format!("Stream init error: {e}")))?;

    let mut final_text = String::new();
    let mut final_steps = 0u32;
    let mut final_input_tokens = 0u32;
    let mut final_output_tokens = 0u32;

    while let Some(event) = stream.next().await {
        match event {
            Ok(gauss_core::agent::AgentStreamEvent::TextDelta { step, delta }) => {
                let event_json = serde_json::to_string(&json!({
                    "type": "text_delta",
                    "step": step,
                    "delta": delta,
                }))
                .unwrap_or_default();
                let _ = stream_callback.call(
                    event_json,
                    napi::threadsafe_function::ThreadsafeFunctionCallMode::NonBlocking,
                );
            }
            Ok(gauss_core::agent::AgentStreamEvent::StepStart { step }) => {
                let event_json = serde_json::to_string(&json!({
                    "type": "step_start",
                    "step": step,
                }))
                .unwrap_or_default();
                let _ = stream_callback.call(
                    event_json,
                    napi::threadsafe_function::ThreadsafeFunctionCallMode::NonBlocking,
                );
            }
            Ok(gauss_core::agent::AgentStreamEvent::ToolCallDelta { step, index }) => {
                let event_json = serde_json::to_string(&json!({
                    "type": "tool_call_delta",
                    "step": step,
                    "index": index,
                }))
                .unwrap_or_default();
                let _ = stream_callback.call(
                    event_json,
                    napi::threadsafe_function::ThreadsafeFunctionCallMode::NonBlocking,
                );
            }
            Ok(gauss_core::agent::AgentStreamEvent::ToolResult {
                step,
                tool_name,
                result,
                is_error,
            }) => {
                let event_json = serde_json::to_string(&json!({
                    "type": "tool_result",
                    "step": step,
                    "toolName": tool_name,
                    "result": result,
                    "isError": is_error,
                }))
                .unwrap_or_default();
                let _ = stream_callback.call(
                    event_json,
                    napi::threadsafe_function::ThreadsafeFunctionCallMode::NonBlocking,
                );
            }
            Ok(gauss_core::agent::AgentStreamEvent::StepFinish {
                step,
                finish_reason,
                has_tool_calls,
            }) => {
                let event_json = serde_json::to_string(&json!({
                    "type": "step_finish",
                    "step": step,
                    "finishReason": format!("{:?}", finish_reason),
                    "hasToolCalls": has_tool_calls,
                }))
                .unwrap_or_default();
                let _ = stream_callback.call(
                    event_json,
                    napi::threadsafe_function::ThreadsafeFunctionCallMode::NonBlocking,
                );
            }
            Ok(gauss_core::agent::AgentStreamEvent::Done { text, steps, usage }) => {
                final_text = text;
                final_steps = steps as u32;
                final_input_tokens = usage.input_tokens as u32;
                final_output_tokens = usage.output_tokens as u32;

                let event_json = serde_json::to_string(&json!({
                    "type": "done",
                    "text": final_text,
                    "steps": final_steps,
                    "inputTokens": final_input_tokens,
                    "outputTokens": final_output_tokens,
                }))
                .unwrap_or_default();
                let _ = stream_callback.call(
                    event_json,
                    napi::threadsafe_function::ThreadsafeFunctionCallMode::NonBlocking,
                );
            }
            Ok(gauss_core::agent::AgentStreamEvent::RawEvent { step, event }) => {
                let event_json = serde_json::to_string(&json!({
                    "type": "raw_event",
                    "step": step,
                    "event": format!("{:?}", event),
                }))
                .unwrap_or_default();
                let _ = stream_callback.call(
                    event_json,
                    napi::threadsafe_function::ThreadsafeFunctionCallMode::NonBlocking,
                );
            }
            Err(e) => {
                let event_json = serde_json::to_string(&json!({
                    "type": "error",
                    "error": format!("{e}"),
                }))
                .unwrap_or_default();
                let _ = stream_callback.call(
                    event_json,
                    napi::threadsafe_function::ThreadsafeFunctionCallMode::NonBlocking,
                );
                return Err(napi::Error::from_reason(format!("Stream error: {e}")));
            }
        }
    }

    Ok(AgentResult {
        text: final_text,
        steps: final_steps,
        input_tokens: final_input_tokens,
        output_tokens: final_output_tokens,
        structured_output: None,
    })
}

// ============ Direct Provider Call ============

/// Call a provider directly (without agent loop).
#[napi]
pub async fn generate(
    provider_handle: u32,
    messages: Vec<JsMessage>,
    temperature: Option<f64>,
    max_tokens: Option<u32>,
) -> Result<serde_json::Value> {
    let provider = get_provider(provider_handle)?;
    let rust_msgs: Vec<RustMessage> = messages.iter().map(js_message_to_rust).collect();

    let opts = GenerateOptions {
        temperature,
        max_tokens,
        ..GenerateOptions::default()
    };

    let result = provider
        .generate(&rust_msgs, &[], &opts)
        .await
        .map_err(|e| napi::Error::from_reason(format!("Generate error: {e}")))?;

    let text = result.text().unwrap_or("").to_string();

    Ok(json!({
        "text": text,
        "usage": {
            "inputTokens": result.usage.input_tokens,
            "outputTokens": result.usage.output_tokens,
        },
        "finishReason": format!("{:?}", result.finish_reason),
    }))
}

/// Call a provider with tool definitions. Returns tool calls if the model requests them.
#[napi]
pub async fn generate_with_tools(
    provider_handle: u32,
    messages: Vec<JsMessage>,
    tools: Vec<ToolDef>,
    temperature: Option<f64>,
    max_tokens: Option<u32>,
) -> Result<serde_json::Value> {
    let provider = get_provider(provider_handle)?;
    let rust_msgs: Vec<RustMessage> = messages.iter().map(js_message_to_rust).collect();

    let rust_tools: Vec<RustTool> = tools
        .iter()
        .map(|td| {
            let mut tb = RustTool::builder(&td.name, &td.description);
            if let Some(ref params) = td.parameters {
                tb = tb.parameters_json(params.clone());
            }
            tb.build()
        })
        .collect();

    let opts = GenerateOptions {
        temperature,
        max_tokens,
        ..GenerateOptions::default()
    };

    let result = provider
        .generate(&rust_msgs, &rust_tools, &opts)
        .await
        .map_err(|e| napi::Error::from_reason(format!("Generate error: {e}")))?;

    let text = result.text().unwrap_or("").to_string();
    let tool_calls: Vec<serde_json::Value> = result
        .message
        .tool_calls()
        .into_iter()
        .map(|(id, name, args)| {
            json!({
                "id": id,
                "name": name,
                "args": args,
            })
        })
        .collect();

    Ok(json!({
        "text": text,
        "toolCalls": tool_calls,
        "usage": {
            "inputTokens": result.usage.input_tokens,
            "outputTokens": result.usage.output_tokens,
        },
        "finishReason": format!("{:?}", result.finish_reason),
    }))
}

// ============ Handle Registries ============

fn memory_registry() -> &'static Mutex<HashMap<u32, Arc<memory::InMemoryMemory>>> {
    use std::sync::OnceLock;
    static R: OnceLock<Mutex<HashMap<u32, Arc<memory::InMemoryMemory>>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

fn network_registry() -> &'static Mutex<HashMap<u32, Arc<tokio::sync::Mutex<network::AgentNetwork>>>>
{
    use std::sync::OnceLock;
    static R: OnceLock<Mutex<HashMap<u32, Arc<tokio::sync::Mutex<network::AgentNetwork>>>>> =
        OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

fn mcp_server_registry() -> &'static Mutex<HashMap<u32, Arc<tokio::sync::Mutex<mcp::McpServer>>>> {
    use std::sync::OnceLock;
    static R: OnceLock<Mutex<HashMap<u32, Arc<tokio::sync::Mutex<mcp::McpServer>>>>> =
        OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

fn middleware_registry() -> &'static Mutex<HashMap<u32, Arc<Mutex<middleware::MiddlewareChain>>>> {
    use std::sync::OnceLock;
    static R: OnceLock<Mutex<HashMap<u32, Arc<Mutex<middleware::MiddlewareChain>>>>> =
        OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

fn approval_registry() -> &'static Mutex<HashMap<u32, Arc<Mutex<hitl::ApprovalManager>>>> {
    use std::sync::OnceLock;
    static R: OnceLock<Mutex<HashMap<u32, Arc<Mutex<hitl::ApprovalManager>>>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

fn checkpoint_registry() -> &'static Mutex<HashMap<u32, Arc<hitl::InMemoryCheckpointStore>>> {
    use std::sync::OnceLock;
    static R: OnceLock<Mutex<HashMap<u32, Arc<hitl::InMemoryCheckpointStore>>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

fn eval_registry() -> &'static Mutex<HashMap<u32, Arc<Mutex<eval::EvalRunner>>>> {
    use std::sync::OnceLock;
    static R: OnceLock<Mutex<HashMap<u32, Arc<Mutex<eval::EvalRunner>>>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

fn telemetry_registry() -> &'static Mutex<HashMap<u32, Arc<Mutex<telemetry::TelemetryCollector>>>> {
    use std::sync::OnceLock;
    static R: OnceLock<Mutex<HashMap<u32, Arc<Mutex<telemetry::TelemetryCollector>>>>> =
        OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

fn vector_store_registry() -> &'static Mutex<HashMap<u32, Arc<rag::InMemoryVectorStore>>> {
    use std::sync::OnceLock;
    static R: OnceLock<Mutex<HashMap<u32, Arc<rag::InMemoryVectorStore>>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

fn next_handle() -> u32 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

// ============ Memory ============

/// Create an in-memory memory store. Returns handle.
#[napi]
pub fn create_memory() -> u32 {
    let id = next_handle();
    memory_registry()
        .lock()
        .unwrap()
        .insert(id, Arc::new(memory::InMemoryMemory::new()));
    id
}

/// Store a memory entry. entry_json must be a MemoryEntry JSON.
#[napi]
pub async fn memory_store(handle: u32, entry_json: String) -> Result<()> {
    use memory::Memory;
    let mem = memory_registry()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| napi::Error::from_reason("Memory not found"))?;
    let entry: memory::MemoryEntry = serde_json::from_str(&entry_json)
        .map_err(|e| napi::Error::from_reason(format!("Invalid entry: {e}")))?;
    mem.store(entry)
        .await
        .map_err(|e| napi::Error::from_reason(format!("{e}")))
}

/// Recall memory entries matching options.
#[napi]
pub async fn memory_recall(handle: u32, options_json: Option<String>) -> Result<serde_json::Value> {
    use memory::Memory;
    let mem = memory_registry()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| napi::Error::from_reason("Memory not found"))?;
    let opts: memory::RecallOptions = match options_json {
        Some(j) => serde_json::from_str(&j)
            .map_err(|e| napi::Error::from_reason(format!("Invalid options: {e}")))?,
        None => memory::RecallOptions::default(),
    };
    let entries = mem
        .recall(opts)
        .await
        .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
    serde_json::to_value(&entries).map_err(|e| napi::Error::from_reason(format!("{e}")))
}

/// Clear memory entries, optionally filtered by session_id.
#[napi]
pub async fn memory_clear(handle: u32, session_id: Option<String>) -> Result<()> {
    let mem = memory_registry()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| napi::Error::from_reason("Memory not found"))?;
    memory::Memory::clear(&*mem, session_id.as_deref())
        .await
        .map_err(|e| napi::Error::from_reason(format!("{e}")))
}

/// Get memory statistics.
#[napi]
pub async fn memory_stats(handle: u32) -> Result<serde_json::Value> {
    use memory::Memory;
    let mem = memory_registry()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| napi::Error::from_reason("Memory not found"))?;
    let stats = mem
        .stats()
        .await
        .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
    serde_json::to_value(&stats).map_err(|e| napi::Error::from_reason(format!("{e}")))
}

#[napi]
pub fn destroy_memory(handle: u32) -> Result<()> {
    memory_registry()
        .lock()
        .unwrap()
        .remove(&handle)
        .ok_or_else(|| napi::Error::from_reason("Memory not found"))?;
    Ok(())
}

// ============ Context ============

/// Precise token count for text (tiktoken cl100k_base).
#[napi]
pub fn count_tokens(text: String) -> u32 {
    context::count_tokens(&text) as u32
}

/// Token count for a specific model's encoding.
#[napi]
pub fn count_tokens_for_model(text: String, model: String) -> u32 {
    context::count_tokens_for_model(&text, &model) as u32
}

/// Token count for an array of messages.
#[napi]
pub fn count_message_tokens(messages: Vec<JsMessage>) -> u32 {
    let msgs: Vec<RustMessage> = messages.iter().map(js_message_to_rust).collect();
    context::count_messages_tokens(&msgs) as u32
}

/// Get context window size for a model.
#[napi]
pub fn get_context_window_size(model: String) -> u32 {
    context::context_window_size(&model) as u32
}

// ============ RAG ============

/// Create an in-memory vector store. Returns handle.
#[napi]
pub fn create_vector_store() -> u32 {
    let id = next_handle();
    vector_store_registry()
        .lock()
        .unwrap()
        .insert(id, Arc::new(rag::InMemoryVectorStore::new()));
    id
}

/// Upsert chunks into the vector store.
#[napi]
pub async fn vector_store_upsert(handle: u32, chunks_json: String) -> Result<()> {
    use rag::VectorStore;
    let store = vector_store_registry()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| napi::Error::from_reason("VectorStore not found"))?;
    let chunks: Vec<rag::Chunk> = serde_json::from_str(&chunks_json)
        .map_err(|e| napi::Error::from_reason(format!("Invalid chunks: {e}")))?;
    store
        .upsert(chunks)
        .await
        .map_err(|e| napi::Error::from_reason(format!("{e}")))
}

/// Search the vector store by embedding vector.
#[napi]
pub async fn vector_store_search(
    handle: u32,
    embedding_json: String,
    top_k: u32,
) -> Result<serde_json::Value> {
    use rag::VectorStore;
    let store = vector_store_registry()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| napi::Error::from_reason("VectorStore not found"))?;
    let embedding: Vec<f32> = serde_json::from_str(&embedding_json)
        .map_err(|e| napi::Error::from_reason(format!("Invalid embedding: {e}")))?;
    let results = store
        .search(&embedding, top_k as usize)
        .await
        .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
    serde_json::to_value(&results).map_err(|e| napi::Error::from_reason(format!("{e}")))
}

#[napi]
pub fn destroy_vector_store(handle: u32) -> Result<()> {
    vector_store_registry()
        .lock()
        .unwrap()
        .remove(&handle)
        .ok_or_else(|| napi::Error::from_reason("VectorStore not found"))?;
    Ok(())
}

/// Compute cosine similarity between two vectors.
#[napi]
pub fn cosine_similarity(a: Vec<f64>, b: Vec<f64>) -> f64 {
    let af: Vec<f32> = a.iter().map(|&x| x as f32).collect();
    let bf: Vec<f32> = b.iter().map(|&x| x as f32).collect();
    rag::cosine_similarity(&af, &bf) as f64
}

// ============ MCP ============

/// Create an MCP server. Returns handle.
#[napi]
pub fn create_mcp_server(name: String, version_str: String) -> u32 {
    let id = next_handle();
    mcp_server_registry().lock().unwrap().insert(
        id,
        Arc::new(tokio::sync::Mutex::new(mcp::McpServer::new(
            name,
            version_str,
        ))),
    );
    id
}

/// Add a tool to the MCP server. tool_json is a ToolDef (gauss tool) JSON.
#[napi]
pub fn mcp_server_add_tool(handle: u32, tool_json: String) -> Result<()> {
    let mcp_tool: mcp::McpTool = serde_json::from_str(&tool_json)
        .map_err(|e| napi::Error::from_reason(format!("Invalid tool: {e}")))?;
    let gauss_tool = mcp::mcp_tool_to_gauss(&mcp_tool);
    let reg = mcp_server_registry();
    let guard = reg.lock().unwrap();
    let server = guard
        .get(&handle)
        .ok_or_else(|| napi::Error::from_reason("McpServer not found"))?;
    server.blocking_lock().add_tool(gauss_tool);
    Ok(())
}

/// Handle an incoming JSON-RPC message. Returns the response JSON.
#[napi]
pub async fn mcp_server_handle(handle: u32, message_json: String) -> Result<serde_json::Value> {
    let server = mcp_server_registry()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| napi::Error::from_reason("McpServer not found"))?;
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
    mcp_server_registry()
        .lock()
        .unwrap()
        .remove(&handle)
        .ok_or_else(|| napi::Error::from_reason("McpServer not found"))?;
    Ok(())
}

// ============ Network ============

/// Create an agent network. Returns handle.
#[napi]
pub fn create_network() -> u32 {
    let id = next_handle();
    network_registry().lock().unwrap().insert(
        id,
        Arc::new(tokio::sync::Mutex::new(network::AgentNetwork::new())),
    );
    id
}

/// Add an agent to the network.
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

    let reg = network_registry();
    let guard = reg.lock().unwrap();
    let net = guard
        .get(&handle)
        .ok_or_else(|| napi::Error::from_reason("Network not found"))?;
    net.blocking_lock().add_agent(node);
    Ok(())
}

/// Set the supervisor agent for the network.
#[napi]
pub fn network_set_supervisor(handle: u32, agent_name: String) -> Result<()> {
    let reg = network_registry();
    let guard = reg.lock().unwrap();
    let net = guard
        .get(&handle)
        .ok_or_else(|| napi::Error::from_reason("Network not found"))?;
    net.blocking_lock().set_supervisor(agent_name);
    Ok(())
}

/// Delegate a task to a specific agent in the network.
#[napi]
pub async fn network_delegate(
    handle: u32,
    agent_name: String,
    prompt: String,
) -> Result<serde_json::Value> {
    let net = network_registry()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| napi::Error::from_reason("Network not found"))?;
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

/// Get all agent cards in the network.
#[napi]
pub fn network_agent_cards(handle: u32) -> Result<serde_json::Value> {
    let reg = network_registry();
    let guard = reg.lock().unwrap();
    let net = guard
        .get(&handle)
        .ok_or_else(|| napi::Error::from_reason("Network not found"))?;
    let net_guard = net.blocking_lock();
    let cards = net_guard.agent_cards();
    serde_json::to_value(&cards).map_err(|e| napi::Error::from_reason(format!("{e}")))
}

#[napi]
pub fn destroy_network(handle: u32) -> Result<()> {
    network_registry()
        .lock()
        .unwrap()
        .remove(&handle)
        .ok_or_else(|| napi::Error::from_reason("Network not found"))?;
    Ok(())
}

// ============ Middleware ============

/// Create a middleware chain. Returns handle.
#[napi]
pub fn create_middleware_chain() -> u32 {
    let id = next_handle();
    middleware_registry()
        .lock()
        .unwrap()
        .insert(id, Arc::new(Mutex::new(middleware::MiddlewareChain::new())));
    id
}

/// Add logging middleware to the chain.
#[napi]
pub fn middleware_use_logging(handle: u32) -> Result<()> {
    let reg = middleware_registry();
    let guard = reg.lock().unwrap();
    let chain = guard
        .get(&handle)
        .ok_or_else(|| napi::Error::from_reason("MiddlewareChain not found"))?;
    chain
        .lock()
        .unwrap()
        .use_middleware(Arc::new(middleware::LoggingMiddleware));
    Ok(())
}

/// Add caching middleware to the chain with a TTL in milliseconds.
#[napi]
pub fn middleware_use_caching(handle: u32, ttl_ms: u32) -> Result<()> {
    let reg = middleware_registry();
    let guard = reg.lock().unwrap();
    let chain = guard
        .get(&handle)
        .ok_or_else(|| napi::Error::from_reason("MiddlewareChain not found"))?;
    chain
        .lock()
        .unwrap()
        .use_middleware(Arc::new(middleware::CachingMiddleware::new(ttl_ms as u64)));
    Ok(())
}

#[napi]
pub fn destroy_middleware_chain(handle: u32) -> Result<()> {
    middleware_registry()
        .lock()
        .unwrap()
        .remove(&handle)
        .ok_or_else(|| napi::Error::from_reason("MiddlewareChain not found"))?;
    Ok(())
}

// ============ HITL ============

/// Create an approval manager. Returns handle.
#[napi]
pub fn create_approval_manager() -> u32 {
    let id = next_handle();
    approval_registry()
        .lock()
        .unwrap()
        .insert(id, Arc::new(Mutex::new(hitl::ApprovalManager::new())));
    id
}

/// Request approval for a tool call. Returns the request ID.
#[napi]
pub fn approval_request(
    handle: u32,
    tool_name: String,
    args_json: String,
    session_id: String,
) -> Result<String> {
    let mgr = approval_registry()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| napi::Error::from_reason("ApprovalManager not found"))?;
    let args: serde_json::Value = serde_json::from_str(&args_json)
        .map_err(|e| napi::Error::from_reason(format!("Invalid args: {e}")))?;
    let req = mgr
        .lock()
        .unwrap()
        .request_approval(tool_name, args, 0, session_id)
        .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
    Ok(req.id.clone())
}

/// Approve a pending request, optionally with modified args.
#[napi]
pub fn approval_approve(
    handle: u32,
    request_id: String,
    modified_args: Option<String>,
) -> Result<()> {
    let mgr = approval_registry()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| napi::Error::from_reason("ApprovalManager not found"))?;
    let args: Option<serde_json::Value> = match modified_args {
        Some(j) => Some(
            serde_json::from_str(&j)
                .map_err(|e| napi::Error::from_reason(format!("Invalid args: {e}")))?,
        ),
        None => None,
    };
    mgr.lock()
        .unwrap()
        .approve(&request_id, args)
        .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
    Ok(())
}

/// Deny a pending request with an optional reason.
#[napi]
pub fn approval_deny(handle: u32, request_id: String, reason: Option<String>) -> Result<()> {
    let mgr = approval_registry()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| napi::Error::from_reason("ApprovalManager not found"))?;
    mgr.lock()
        .unwrap()
        .deny(&request_id, reason)
        .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
    Ok(())
}

/// List all pending approval requests.
#[napi]
pub fn approval_list_pending(handle: u32) -> Result<serde_json::Value> {
    let mgr = approval_registry()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| napi::Error::from_reason("ApprovalManager not found"))?;
    let pending = mgr
        .lock()
        .unwrap()
        .list_pending()
        .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
    serde_json::to_value(&pending).map_err(|e| napi::Error::from_reason(format!("{e}")))
}

#[napi]
pub fn destroy_approval_manager(handle: u32) -> Result<()> {
    approval_registry()
        .lock()
        .unwrap()
        .remove(&handle)
        .ok_or_else(|| napi::Error::from_reason("ApprovalManager not found"))?;
    Ok(())
}

// ============ Checkpoint Store ============

/// Create an in-memory checkpoint store. Returns handle.
#[napi]
pub fn create_checkpoint_store() -> u32 {
    let id = next_handle();
    checkpoint_registry()
        .lock()
        .unwrap()
        .insert(id, Arc::new(hitl::InMemoryCheckpointStore::new()));
    id
}

/// Save a checkpoint.
#[napi]
pub async fn checkpoint_save(handle: u32, checkpoint_json: String) -> Result<()> {
    use hitl::CheckpointStore;
    let store = checkpoint_registry()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| napi::Error::from_reason("CheckpointStore not found"))?;
    let cp: hitl::Checkpoint = serde_json::from_str(&checkpoint_json)
        .map_err(|e| napi::Error::from_reason(format!("Invalid checkpoint: {e}")))?;
    store
        .save(&cp)
        .await
        .map_err(|e| napi::Error::from_reason(format!("{e}")))
}

/// Load a checkpoint by ID.
#[napi]
pub async fn checkpoint_load(handle: u32, checkpoint_id: String) -> Result<serde_json::Value> {
    use hitl::CheckpointStore;
    let store = checkpoint_registry()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| napi::Error::from_reason("CheckpointStore not found"))?;
    let cp = store
        .load(&checkpoint_id)
        .await
        .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
    serde_json::to_value(&cp).map_err(|e| napi::Error::from_reason(format!("{e}")))
}

/// Load the latest checkpoint for a session.
#[napi]
pub async fn checkpoint_load_latest(handle: u32, session_id: String) -> Result<serde_json::Value> {
    use hitl::CheckpointStore;
    let store = checkpoint_registry()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| napi::Error::from_reason("CheckpointStore not found"))?;
    let cp = store
        .load_latest(&session_id)
        .await
        .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
    serde_json::to_value(&cp).map_err(|e| napi::Error::from_reason(format!("{e}")))
}

#[napi]
pub fn destroy_checkpoint_store(handle: u32) -> Result<()> {
    checkpoint_registry()
        .lock()
        .unwrap()
        .remove(&handle)
        .ok_or_else(|| napi::Error::from_reason("CheckpointStore not found"))?;
    Ok(())
}

// ============ Eval ============

/// Create an eval runner. Returns handle.
#[napi]
pub fn create_eval_runner(threshold: Option<f64>) -> u32 {
    let id = next_handle();
    let mut runner = eval::EvalRunner::new();
    if let Some(t) = threshold {
        runner = runner.with_threshold(t);
    }
    eval_registry()
        .lock()
        .unwrap()
        .insert(id, Arc::new(Mutex::new(runner)));
    id
}

/// Add a built-in scorer: "exact_match", "contains", or "length_ratio".
#[napi]
pub fn eval_add_scorer(handle: u32, scorer_type: String) -> Result<()> {
    let reg = eval_registry();
    let guard = reg.lock().unwrap();
    let runner = guard
        .get(&handle)
        .ok_or_else(|| napi::Error::from_reason("EvalRunner not found"))?;
    let scorer: Arc<dyn eval::Scorer> = match scorer_type.as_str() {
        "exact_match" => Arc::new(eval::ExactMatchScorer),
        "contains" => Arc::new(eval::ContainsScorer),
        "length_ratio" => Arc::new(eval::LengthRatioScorer),
        other => {
            return Err(napi::Error::from_reason(format!(
                "Unknown scorer: {other}. Use: exact_match, contains, length_ratio"
            )));
        }
    };
    runner.lock().unwrap().add_scorer(scorer);
    Ok(())
}

/// Load evaluation cases from a JSONL string.
#[napi]
pub fn load_dataset_jsonl(jsonl: String) -> Result<serde_json::Value> {
    let cases =
        eval::load_dataset_jsonl(&jsonl).map_err(|e| napi::Error::from_reason(format!("{e}")))?;
    serde_json::to_value(&cases).map_err(|e| napi::Error::from_reason(format!("{e}")))
}

/// Load evaluation cases from a JSON array string.
#[napi]
pub fn load_dataset_json(json_str: String) -> Result<serde_json::Value> {
    let cases =
        eval::load_dataset_json(&json_str).map_err(|e| napi::Error::from_reason(format!("{e}")))?;
    serde_json::to_value(&cases).map_err(|e| napi::Error::from_reason(format!("{e}")))
}

#[napi]
pub fn destroy_eval_runner(handle: u32) -> Result<()> {
    eval_registry()
        .lock()
        .unwrap()
        .remove(&handle)
        .ok_or_else(|| napi::Error::from_reason("EvalRunner not found"))?;
    Ok(())
}

// ============ Telemetry ============

/// Create a telemetry collector. Returns handle.
#[napi]
pub fn create_telemetry() -> u32 {
    let id = next_handle();
    telemetry_registry().lock().unwrap().insert(
        id,
        Arc::new(Mutex::new(telemetry::TelemetryCollector::new())),
    );
    id
}

/// Record a span (JSON).
#[napi]
pub fn telemetry_record_span(handle: u32, span_json: String) -> Result<()> {
    let reg = telemetry_registry();
    let guard = reg.lock().unwrap();
    let collector = guard
        .get(&handle)
        .ok_or_else(|| napi::Error::from_reason("TelemetryCollector not found"))?;
    let span: telemetry::SpanRecord = serde_json::from_str(&span_json)
        .map_err(|e| napi::Error::from_reason(format!("Invalid span: {e}")))?;
    collector.lock().unwrap().record_span(span);
    Ok(())
}

/// Export all recorded spans.
#[napi]
pub fn telemetry_export_spans(handle: u32) -> Result<serde_json::Value> {
    let reg = telemetry_registry();
    let guard = reg.lock().unwrap();
    let collector = guard
        .get(&handle)
        .ok_or_else(|| napi::Error::from_reason("TelemetryCollector not found"))?;
    let spans = collector.lock().unwrap().export_spans();
    serde_json::to_value(&spans).map_err(|e| napi::Error::from_reason(format!("{e}")))
}

/// Export aggregated agent metrics.
#[napi]
pub fn telemetry_export_metrics(handle: u32) -> Result<serde_json::Value> {
    let reg = telemetry_registry();
    let guard = reg.lock().unwrap();
    let collector = guard
        .get(&handle)
        .ok_or_else(|| napi::Error::from_reason("TelemetryCollector not found"))?;
    let metrics = collector.lock().unwrap().export_metrics();
    serde_json::to_value(&metrics).map_err(|e| napi::Error::from_reason(format!("{e}")))
}

/// Clear all telemetry data.
#[napi]
pub fn telemetry_clear(handle: u32) -> Result<()> {
    let reg = telemetry_registry();
    let guard = reg.lock().unwrap();
    let collector = guard
        .get(&handle)
        .ok_or_else(|| napi::Error::from_reason("TelemetryCollector not found"))?;
    collector.lock().unwrap().clear();
    Ok(())
}

#[napi]
pub fn destroy_telemetry(handle: u32) -> Result<()> {
    telemetry_registry()
        .lock()
        .unwrap()
        .remove(&handle)
        .ok_or_else(|| napi::Error::from_reason("TelemetryCollector not found"))?;
    Ok(())
}

// ============ Guardrails ============

use gauss_core::guardrail;

fn guardrail_chain_registry() -> &'static Mutex<HashMap<u32, guardrail::GuardrailChain>> {
    use std::sync::OnceLock;
    static REG: OnceLock<Mutex<HashMap<u32, guardrail::GuardrailChain>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

#[napi]
pub fn create_guardrail_chain() -> u32 {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    guardrail_chain_registry()
        .lock()
        .unwrap()
        .insert(id, guardrail::GuardrailChain::new());
    id
}

#[napi]
pub fn guardrail_chain_add_content_moderation(
    handle: u32,
    block_patterns: Vec<String>,
    warn_patterns: Vec<String>,
) -> Result<()> {
    let mut reg = guardrail_chain_registry().lock().unwrap();
    let chain = reg
        .get_mut(&handle)
        .ok_or_else(|| napi::Error::from_reason("GuardrailChain not found"))?;
    let mut g = guardrail::ContentModerationGuardrail::new();
    for p in block_patterns {
        g = g.block_pattern(&p, format!("Blocked pattern: {p}"));
    }
    for p in warn_patterns {
        g = g.warn_pattern(&p, format!("Warning pattern: {p}"));
    }
    chain.add(Arc::new(g));
    Ok(())
}

#[napi]
pub fn guardrail_chain_add_pii_detection(handle: u32, action: String) -> Result<()> {
    let pii_action = match action.as_str() {
        "block" => guardrail::PiiAction::Block,
        "warn" => guardrail::PiiAction::Warn,
        "redact" => guardrail::PiiAction::Redact,
        _ => {
            return Err(napi::Error::from_reason(
                "Invalid PII action: block|warn|redact",
            ));
        }
    };
    let mut reg = guardrail_chain_registry().lock().unwrap();
    let chain = reg
        .get_mut(&handle)
        .ok_or_else(|| napi::Error::from_reason("GuardrailChain not found"))?;
    chain.add(Arc::new(guardrail::PiiDetectionGuardrail::new(pii_action)));
    Ok(())
}

#[napi]
pub fn guardrail_chain_add_token_limit(
    handle: u32,
    max_input: Option<u32>,
    max_output: Option<u32>,
) -> Result<()> {
    let mut g = guardrail::TokenLimitGuardrail::new();
    if let Some(m) = max_input {
        g = g.max_input(m as usize);
    }
    if let Some(m) = max_output {
        g = g.max_output(m as usize);
    }
    let mut reg = guardrail_chain_registry().lock().unwrap();
    let chain = reg
        .get_mut(&handle)
        .ok_or_else(|| napi::Error::from_reason("GuardrailChain not found"))?;
    chain.add(Arc::new(g));
    Ok(())
}

#[napi]
pub fn guardrail_chain_add_regex_filter(
    handle: u32,
    block_rules: Vec<String>,
    warn_rules: Vec<String>,
) -> Result<()> {
    let mut g = guardrail::RegexFilterGuardrail::new();
    for r in block_rules {
        g = g.block(&r, format!("Blocked by regex: {r}"));
    }
    for r in warn_rules {
        g = g.warn(&r, format!("Warning by regex: {r}"));
    }
    let mut reg = guardrail_chain_registry().lock().unwrap();
    let chain = reg
        .get_mut(&handle)
        .ok_or_else(|| napi::Error::from_reason("GuardrailChain not found"))?;
    chain.add(Arc::new(g));
    Ok(())
}

#[napi]
pub fn guardrail_chain_add_schema(handle: u32, schema_json: String) -> Result<()> {
    let schema: serde_json::Value = serde_json::from_str(&schema_json)
        .map_err(|e| napi::Error::from_reason(format!("Invalid JSON schema: {e}")))?;
    let mut reg = guardrail_chain_registry().lock().unwrap();
    let chain = reg
        .get_mut(&handle)
        .ok_or_else(|| napi::Error::from_reason("GuardrailChain not found"))?;
    chain.add(Arc::new(guardrail::SchemaGuardrail::new(schema)));
    Ok(())
}

#[napi]
pub fn guardrail_chain_list(handle: u32) -> Result<Vec<String>> {
    let reg = guardrail_chain_registry().lock().unwrap();
    let chain = reg
        .get(&handle)
        .ok_or_else(|| napi::Error::from_reason("GuardrailChain not found"))?;
    Ok(chain.list().into_iter().map(String::from).collect())
}

#[napi]
pub fn destroy_guardrail_chain(handle: u32) -> Result<()> {
    guardrail_chain_registry()
        .lock()
        .unwrap()
        .remove(&handle)
        .ok_or_else(|| napi::Error::from_reason("GuardrailChain not found"))?;
    Ok(())
}

// ============ Resilience ============

use gauss_core::resilience;

#[napi]
pub fn create_fallback_provider(provider_handles: Vec<u32>) -> Result<u32> {
    let prov_reg = providers().lock().unwrap();
    let mut providers_vec: Vec<Arc<dyn Provider>> = Vec::new();
    for h in provider_handles {
        let p = prov_reg
            .get(&h)
            .ok_or_else(|| napi::Error::from_reason(format!("Provider {h} not found")))?
            .clone();
        providers_vec.push(p);
    }
    drop(prov_reg);

    let fallback = Arc::new(resilience::FallbackProvider::new(providers_vec));
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    providers().lock().unwrap().insert(id, fallback);
    Ok(id)
}

#[napi]
pub fn create_circuit_breaker(
    provider_handle: u32,
    failure_threshold: Option<u32>,
    recovery_timeout_ms: Option<u32>,
) -> Result<u32> {
    let inner = providers()
        .lock()
        .unwrap()
        .get(&provider_handle)
        .ok_or_else(|| napi::Error::from_reason("Provider not found"))?
        .clone();

    let config = resilience::CircuitBreakerConfig {
        failure_threshold: failure_threshold.unwrap_or(5),
        recovery_timeout_ms: recovery_timeout_ms.map(|v| v as u64).unwrap_or(30_000),
        success_threshold: 1,
    };

    let cb = Arc::new(resilience::CircuitBreaker::new(inner, config));
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    providers().lock().unwrap().insert(id, cb);
    Ok(id)
}

#[napi]
pub fn create_resilient_provider(
    primary_handle: u32,
    fallback_handles: Vec<u32>,
    enable_circuit_breaker: Option<bool>,
) -> Result<u32> {
    let prov_reg = providers().lock().unwrap();
    let primary = prov_reg
        .get(&primary_handle)
        .ok_or_else(|| napi::Error::from_reason("Primary provider not found"))?
        .clone();

    let mut builder = resilience::ResilientProviderBuilder::new(primary);
    builder = builder.retry(RetryConfig::default());

    if enable_circuit_breaker.unwrap_or(false) {
        builder = builder.circuit_breaker(resilience::CircuitBreakerConfig::default());
    }

    for h in &fallback_handles {
        let fb = prov_reg
            .get(h)
            .ok_or_else(|| napi::Error::from_reason(format!("Fallback provider {h} not found")))?
            .clone();
        builder = builder.fallback(fb);
    }
    drop(prov_reg);

    let provider = builder.build();
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    providers().lock().unwrap().insert(id, provider);
    Ok(id)
}

// ============ Stream Transform ============

use gauss_core::stream_transform;

#[napi]
pub fn parse_partial_json(text: String) -> Option<String> {
    stream_transform::parse_partial_json(&text).map(|v| v.to_string())
}

// ============ Plugin System ============

use gauss_core::plugin;

fn plugin_registry_reg() -> &'static Mutex<HashMap<u32, plugin::PluginRegistry>> {
    use std::sync::OnceLock;
    static REG: OnceLock<Mutex<HashMap<u32, plugin::PluginRegistry>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

#[napi]
pub fn create_plugin_registry() -> u32 {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    plugin_registry_reg()
        .lock()
        .unwrap()
        .insert(id, plugin::PluginRegistry::new());
    id
}

#[napi]
pub fn plugin_registry_add_telemetry(handle: u32) -> Result<()> {
    let mut reg = plugin_registry_reg().lock().unwrap();
    let registry = reg
        .get_mut(&handle)
        .ok_or_else(|| napi::Error::from_reason("PluginRegistry not found"))?;
    registry.register(Arc::new(plugin::TelemetryPlugin));
    Ok(())
}

#[napi]
pub fn plugin_registry_add_memory(handle: u32) -> Result<()> {
    let mut reg = plugin_registry_reg().lock().unwrap();
    let registry = reg
        .get_mut(&handle)
        .ok_or_else(|| napi::Error::from_reason("PluginRegistry not found"))?;
    registry.register(Arc::new(plugin::MemoryPlugin));
    Ok(())
}

#[napi]
pub fn plugin_registry_list(handle: u32) -> Result<Vec<String>> {
    let reg = plugin_registry_reg().lock().unwrap();
    let registry = reg
        .get(&handle)
        .ok_or_else(|| napi::Error::from_reason("PluginRegistry not found"))?;
    Ok(registry.list().into_iter().map(String::from).collect())
}

#[napi]
pub fn plugin_registry_emit(handle: u32, event_json: String) -> Result<()> {
    let event: plugin::GaussEvent = serde_json::from_str(&event_json)
        .map_err(|e| napi::Error::from_reason(format!("Invalid event JSON: {e}")))?;
    let reg = plugin_registry_reg().lock().unwrap();
    let registry = reg
        .get(&handle)
        .ok_or_else(|| napi::Error::from_reason("PluginRegistry not found"))?;
    registry.emit(&event);
    Ok(())
}

#[napi]
pub fn destroy_plugin_registry(handle: u32) -> Result<()> {
    plugin_registry_reg()
        .lock()
        .unwrap()
        .remove(&handle)
        .ok_or_else(|| napi::Error::from_reason("PluginRegistry not found"))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tool Validator (patterns module)
// ---------------------------------------------------------------------------

use gauss_core::patterns::{CoercionStrategy, ToolValidator as RustToolValidator};
use std::sync::OnceLock;

fn tool_validator_reg() -> &'static Mutex<HashMap<u32, RustToolValidator>> {
    static REG: OnceLock<Mutex<HashMap<u32, RustToolValidator>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

#[napi]
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
    tool_validator_reg().lock().unwrap().insert(id, validator);
    id
}

#[napi]
pub fn tool_validator_validate(handle: u32, input: String, schema: String) -> Result<String> {
    let reg = tool_validator_reg().lock().unwrap();
    let validator = reg
        .get(&handle)
        .ok_or_else(|| napi::Error::from_reason("ToolValidator not found"))?;
    let input_val: serde_json::Value =
        serde_json::from_str(&input).map_err(|e| napi::Error::from_reason(format!("{e}")))?;
    let schema_val: serde_json::Value =
        serde_json::from_str(&schema).map_err(|e| napi::Error::from_reason(format!("{e}")))?;
    let result = validator
        .validate(input_val, &schema_val)
        .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
    serde_json::to_string(&result).map_err(|e| napi::Error::from_reason(format!("{e}")))
}

#[napi]
pub fn destroy_tool_validator(handle: u32) -> Result<()> {
    tool_validator_reg()
        .lock()
        .unwrap()
        .remove(&handle)
        .ok_or_else(|| napi::Error::from_reason("ToolValidator not found"))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Agent Config (config module)
// ---------------------------------------------------------------------------

#[napi]
pub fn agent_config_from_json(json_str: String) -> Result<String> {
    let config = gauss_core::config::AgentConfig::from_json(&json_str)
        .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
    config
        .to_json()
        .map_err(|e| napi::Error::from_reason(format!("{e}")))
}

#[napi]
pub fn agent_config_resolve_env(value: String) -> String {
    gauss_core::config::resolve_env(&value)
}

// =============================================================================
// Graph API — handle-based DAG builder and executor
// =============================================================================

fn graphs() -> &'static Mutex<HashMap<u32, GraphState>> {
    use std::sync::OnceLock;
    static GRAPHS: OnceLock<Mutex<HashMap<u32, GraphState>>> = OnceLock::new();
    GRAPHS.get_or_init(|| Mutex::new(HashMap::new()))
}

struct GraphState {
    nodes: Vec<GraphNodeDef>,
    edges: Vec<(String, String)>,
}

struct GraphNodeDef {
    id: String,
    agent_name: String,
    provider_handle: u32,
    instructions: Option<String>,
    tools: Vec<(String, String, Option<serde_json::Value>)>,
}

#[napi]
pub fn create_graph() -> u32 {
    let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
    graphs().lock().unwrap().insert(
        id,
        GraphState {
            nodes: vec![],
            edges: vec![],
        },
    );
    id
}

#[napi]
pub fn graph_add_node(
    handle: u32,
    node_id: String,
    agent_name: String,
    provider_handle: u32,
    instructions: Option<String>,
    tools: Vec<ToolDef>,
) -> Result<()> {
    let mut g = graphs().lock().unwrap();
    let state = g
        .get_mut(&handle)
        .ok_or_else(|| napi::Error::from_reason("Graph not found"))?;
    state.nodes.push(GraphNodeDef {
        id: node_id,
        agent_name,
        provider_handle,
        instructions,
        tools: tools
            .into_iter()
            .map(|t| (t.name, t.description, t.parameters))
            .collect(),
    });
    Ok(())
}

#[napi]
pub fn graph_add_edge(handle: u32, from: String, to: String) -> Result<()> {
    let mut g = graphs().lock().unwrap();
    let state = g
        .get_mut(&handle)
        .ok_or_else(|| napi::Error::from_reason("Graph not found"))?;
    state.edges.push((from, to));
    Ok(())
}

#[napi]
pub async fn graph_run(handle: u32, prompt: String) -> Result<serde_json::Value> {
    let state = {
        let g = graphs().lock().unwrap();
        let s = g
            .get(&handle)
            .ok_or_else(|| napi::Error::from_reason("Graph not found"))?;
        // Clone node configs for async context
        let nodes: Vec<_> = s
            .nodes
            .iter()
            .map(|n| {
                (
                    n.id.clone(),
                    n.agent_name.clone(),
                    n.provider_handle,
                    n.instructions.clone(),
                    n.tools.clone(),
                )
            })
            .collect();
        let edges = s.edges.clone();
        (nodes, edges)
    };

    let (node_defs, edges) = state;

    // Build a gauss_core::Graph
    let mut builder = gauss_core::Graph::builder();

    for (node_id, agent_name, prov_handle, instructions, tools) in &node_defs {
        let provider = {
            let provs = providers().lock().unwrap();
            provs.get(prov_handle).cloned().ok_or_else(|| {
                napi::Error::from_reason(format!("Provider {prov_handle} not found"))
            })?
        };

        let mut agent_builder = RustAgent::builder(agent_name.clone(), provider);
        if let Some(instr) = instructions {
            agent_builder = agent_builder.instructions(instr.clone());
        }
        for (name, desc, params) in tools {
            let mut tb = RustTool::builder(name, desc);
            if let Some(p) = params {
                tb = tb.parameters_json(p.clone());
            }
            agent_builder = agent_builder.tool(tb.build());
        }
        let agent = agent_builder.build();

        let nid = node_id.clone();
        builder = builder.node(nid, agent, |_deps| {
            // Pass completed node outputs as context messages
            vec![RustMessage::user("Continue based on prior context.")]
        });
    }

    for (from, to) in &edges {
        builder = builder.edge(from.clone(), to.clone());
    }

    let graph = builder.build();
    let result = graph
        .run(prompt)
        .await
        .map_err(|e| napi::Error::from_reason(format!("{e}")))?;

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

    Ok(json!({
        "outputs": outputs,
        "final_text": result.final_output.map(|o| o.text),
    }))
}

#[napi]
pub fn destroy_graph(handle: u32) -> Result<()> {
    graphs()
        .lock()
        .unwrap()
        .remove(&handle)
        .ok_or_else(|| napi::Error::from_reason("Graph not found"))?;
    Ok(())
}

// =============================================================================
// Workflow API — handle-based sequential/parallel workflow executor
// =============================================================================

fn workflows() -> &'static Mutex<HashMap<u32, WorkflowState>> {
    use std::sync::OnceLock;
    static WORKFLOWS: OnceLock<Mutex<HashMap<u32, WorkflowState>>> = OnceLock::new();
    WORKFLOWS.get_or_init(|| Mutex::new(HashMap::new()))
}

struct WorkflowState {
    steps: Vec<WorkflowStepDef>,
    dependencies: Vec<(String, String)>,
}

struct WorkflowStepDef {
    id: String,
    agent_name: String,
    provider_handle: u32,
    instructions: Option<String>,
    tools: Vec<(String, String, Option<serde_json::Value>)>,
}

#[napi]
pub fn create_workflow() -> u32 {
    let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
    workflows().lock().unwrap().insert(
        id,
        WorkflowState {
            steps: vec![],
            dependencies: vec![],
        },
    );
    id
}

#[napi]
pub fn workflow_add_step(
    handle: u32,
    step_id: String,
    agent_name: String,
    provider_handle: u32,
    instructions: Option<String>,
    tools: Vec<ToolDef>,
) -> Result<()> {
    let mut w = workflows().lock().unwrap();
    let state = w
        .get_mut(&handle)
        .ok_or_else(|| napi::Error::from_reason("Workflow not found"))?;
    state.steps.push(WorkflowStepDef {
        id: step_id,
        agent_name,
        provider_handle,
        instructions,
        tools: tools
            .into_iter()
            .map(|t| (t.name, t.description, t.parameters))
            .collect(),
    });
    Ok(())
}

#[napi]
pub fn workflow_add_dependency(handle: u32, step_id: String, depends_on: String) -> Result<()> {
    let mut w = workflows().lock().unwrap();
    let state = w
        .get_mut(&handle)
        .ok_or_else(|| napi::Error::from_reason("Workflow not found"))?;
    state.dependencies.push((step_id, depends_on));
    Ok(())
}

#[napi]
pub async fn workflow_run(handle: u32, prompt: String) -> Result<serde_json::Value> {
    let state = {
        let w = workflows().lock().unwrap();
        let s = w
            .get(&handle)
            .ok_or_else(|| napi::Error::from_reason("Workflow not found"))?;
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
        (steps, deps)
    };

    let (step_defs, deps) = state;

    let mut builder = gauss_core::Workflow::builder();

    for (step_id, agent_name, prov_handle, instructions, tools) in &step_defs {
        let provider = {
            let provs = providers().lock().unwrap();
            provs.get(prov_handle).cloned().ok_or_else(|| {
                napi::Error::from_reason(format!("Provider {prov_handle} not found"))
            })?
        };

        let mut agent_builder = RustAgent::builder(agent_name.clone(), provider);
        if let Some(instr) = instructions {
            agent_builder = agent_builder.instructions(instr.clone());
        }
        for (name, desc, params) in tools {
            let mut tb = RustTool::builder(name, desc);
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
    let result = workflow
        .run(initial)
        .await
        .map_err(|e| napi::Error::from_reason(format!("{e}")))?;

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

    Ok(json!({ "steps": outputs }))
}

#[napi]
pub fn destroy_workflow(handle: u32) -> Result<()> {
    workflows()
        .lock()
        .unwrap()
        .remove(&handle)
        .ok_or_else(|| napi::Error::from_reason("Workflow not found"))?;
    Ok(())
}
