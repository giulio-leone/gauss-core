#[macro_use]
extern crate napi_derive;

use gauss_core::agent::{Agent as RustAgent, AgentOutput as RustAgentOutput, StopCondition};
use gauss_core::message::Message as RustMessage;
use gauss_core::provider::anthropic::AnthropicProvider;
use gauss_core::provider::deepseek::DeepSeekProvider;
use gauss_core::provider::google::GoogleProvider;
use gauss_core::provider::groq::GroqProvider;
use gauss_core::provider::ollama::OllamaProvider;
use gauss_core::provider::openai::OpenAiProvider;
use gauss_core::provider::retry::{RetryConfig, RetryProvider};
use gauss_core::provider::{GenerateOptions, Provider, ProviderConfig};
use gauss_core::tool::Tool as RustTool;
use napi::bindgen_prelude::*;
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
