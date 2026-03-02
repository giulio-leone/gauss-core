use crate::registry::{HandleRegistry, py_err};
use gauss_core::provider::anthropic::AnthropicProvider;
use gauss_core::provider::deepseek::DeepSeekProvider;
use gauss_core::provider::fireworks::FireworksProvider;
use gauss_core::provider::google::GoogleProvider;
use gauss_core::provider::groq::GroqProvider;
use gauss_core::provider::mistral::MistralProvider;
use gauss_core::provider::ollama::OllamaProvider;
use gauss_core::provider::openai::OpenAiProvider;
use gauss_core::provider::openrouter::OpenRouterProvider;
use gauss_core::provider::perplexity::PerplexityProvider;
use gauss_core::provider::retry::{RetryConfig, RetryProvider};
use gauss_core::provider::together::TogetherProvider;
use gauss_core::provider::xai::XaiProvider;
use gauss_core::provider::{Provider, ProviderConfig};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde_json::json;
use std::sync::Arc;

pub static PROVIDERS: HandleRegistry<Arc<dyn Provider>> = HandleRegistry::new();

pub fn get_provider(handle: u32) -> PyResult<Arc<dyn Provider>> {
    PROVIDERS.get_clone(handle)
}

/// Create a provider. Returns handle ID.
#[pyfunction]
#[pyo3(signature = (provider_type, model, api_key, base_url=None, max_retries=None))]
pub fn create_provider(
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
        "openrouter" => Arc::new(OpenRouterProvider::create(model, config)),
        "together" => Arc::new(TogetherProvider::create(model, config)),
        "fireworks" => Arc::new(FireworksProvider::create(model, config)),
        "mistral" => Arc::new(MistralProvider::create(model, config)),
        "perplexity" => Arc::new(PerplexityProvider::create(model, config)),
        "xai" => Arc::new(XaiProvider::create(model, config)),
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

    Ok(PROVIDERS.insert(provider))
}

/// Destroy a provider.
#[pyfunction]
pub fn destroy_provider(handle: u32) -> PyResult<()> {
    PROVIDERS.remove(handle)
}

/// Get provider capabilities. Returns JSON string.
#[pyfunction]
pub fn get_provider_capabilities(provider_handle: u32) -> PyResult<String> {
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
    serde_json::to_string(&output).map_err(|e| py_err(format!("Serialize error: {e}")))
}

/// Estimate request cost from token usage for a given model.
#[pyfunction]
#[pyo3(signature = (model, input_tokens, output_tokens, reasoning_tokens=None, cache_read_tokens=None, cache_creation_tokens=None))]
pub fn estimate_cost(
    model: &str,
    input_tokens: u32,
    output_tokens: u32,
    reasoning_tokens: Option<u32>,
    cache_read_tokens: Option<u32>,
    cache_creation_tokens: Option<u32>,
) -> PyResult<String> {
    let usage = gauss_core::message::Usage {
        input_tokens: input_tokens as u64,
        output_tokens: output_tokens as u64,
        reasoning_tokens: reasoning_tokens.map(|v| v as u64),
        cache_read_tokens: cache_read_tokens.map(|v| v as u64),
        cache_creation_tokens: cache_creation_tokens.map(|v| v as u64),
    };
    let estimate = gauss_core::cost::estimate_cost(model, &usage);
    serde_json::to_string(&estimate).map_err(|e| py_err(format!("Serialize error: {e}")))
}
