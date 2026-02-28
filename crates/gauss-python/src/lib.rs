use gauss_core::agent::{Agent as RustAgent, StopCondition};
use gauss_core::message::Message as RustMessage;
use gauss_core::provider::anthropic::AnthropicProvider;
use gauss_core::provider::google::GoogleProvider;
use gauss_core::provider::openai::OpenAiProvider;
use gauss_core::provider::retry::{RetryConfig, RetryProvider};
use gauss_core::provider::{GenerateOptions, Provider, ProviderConfig};
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

/// Gauss Core Python module.
#[pymodule]
#[pyo3(name = "gauss_core")]
fn gauss_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(create_provider, m)?)?;
    m.add_function(wrap_pyfunction!(destroy_provider, m)?)?;
    m.add_function(wrap_pyfunction!(generate, m)?)?;
    m.add_function(wrap_pyfunction!(agent_run, m)?)?;
    Ok(())
}
