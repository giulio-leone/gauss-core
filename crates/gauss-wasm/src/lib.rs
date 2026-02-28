use gauss_core::agent::{Agent as RustAgent, StopCondition};
use gauss_core::message::Message as RustMessage;
use gauss_core::provider::anthropic::AnthropicProvider;
use gauss_core::provider::google::GoogleProvider;
use gauss_core::provider::openai::OpenAiProvider;
use gauss_core::provider::retry::{RetryConfig, RetryProvider};
use gauss_core::provider::{GenerateOptions, Provider, ProviderConfig};
use serde_json::json;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use wasm_bindgen::prelude::*;

fn providers() -> &'static Mutex<HashMap<u32, Arc<dyn Provider>>> {
    static PROVIDERS: OnceLock<Mutex<HashMap<u32, Arc<dyn Provider>>>> = OnceLock::new();
    PROVIDERS.get_or_init(|| Mutex::new(HashMap::new()))
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

    let inner: Arc<dyn Provider> = match provider_type {
        "openai" => Arc::new(OpenAiProvider::new(model, config)),
        "anthropic" => Arc::new(AnthropicProvider::new(model, config)),
        "google" => Arc::new(GoogleProvider::new(model, config)),
        other => return Err(JsValue::from_str(&format!("Unknown provider: {other}"))),
    };

    let provider: Arc<dyn Provider> = Arc::new(RetryProvider::new(
        inner,
        RetryConfig {
            max_retries: 3,
            ..RetryConfig::default()
        },
    ));

    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    providers().lock().unwrap().insert(id, provider);
    Ok(id)
}

/// Destroys a provider.
#[wasm_bindgen(js_name = "destroyProvider")]
pub fn destroy_provider(handle: u32) -> Result<(), JsValue> {
    providers()
        .lock()
        .unwrap()
        .remove(&handle)
        .ok_or_else(|| JsValue::from_str(&format!("Provider {handle} not found")))?;
    Ok(())
}

fn get_provider(handle: u32) -> Result<Arc<dyn Provider>, JsValue> {
    providers()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| JsValue::from_str(&format!("Provider {handle} not found")))
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
