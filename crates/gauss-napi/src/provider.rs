use crate::registry::HandleRegistry;
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
use napi::bindgen_prelude::*;
use serde_json::json;
use std::sync::Arc;

pub static PROVIDERS: HandleRegistry<Arc<dyn Provider>> = HandleRegistry::new();

#[napi(object)]
pub struct ProviderOptions {
    pub api_key: String,
    pub base_url: Option<String>,
    pub timeout_ms: Option<u32>,
    pub max_retries: Option<u32>,
    pub organization: Option<String>,
}

/// Creates a provider and returns its handle ID.
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
        "openrouter" => Arc::new(OpenRouterProvider::create(model, config)),
        "together" => Arc::new(TogetherProvider::create(model, config)),
        "fireworks" => Arc::new(FireworksProvider::create(model, config)),
        "mistral" => Arc::new(MistralProvider::create(model, config)),
        "perplexity" => Arc::new(PerplexityProvider::create(model, config)),
        "xai" => Arc::new(XaiProvider::create(model, config)),
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
                ..Default::default()
            },
        ))
    } else {
        inner
    };

    Ok(PROVIDERS.insert(provider))
}

#[napi]
pub fn destroy_provider(handle: u32) -> Result<()> {
    PROVIDERS.remove(handle)
}

pub fn get_provider(handle: u32) -> Result<Arc<dyn Provider>> {
    PROVIDERS.get_clone(handle)
}

/// Get the capabilities of a provider.
#[napi]
pub fn get_provider_capabilities(provider_handle: u32) -> Result<serde_json::Value> {
    let provider = get_provider(provider_handle)?;
    let caps = provider.capabilities();
    Ok(json!({
        "streaming": caps.streaming,
        "toolUse": caps.tool_use,
        "vision": caps.vision,
        "audio": caps.audio,
        "extendedThinking": caps.extended_thinking,
        "citations": caps.citations,
        "cacheControl": caps.cache_control,
        "structuredOutput": caps.structured_output,
        "reasoningEffort": caps.reasoning_effort,
        "imageGeneration": caps.image_generation,
        "grounding": caps.grounding,
        "codeExecution": caps.code_execution,
        "webSearch": caps.web_search,
    }))
}

/// Estimate request cost from token usage for a given model.
#[napi]
pub fn estimate_cost(
    model: String,
    input_tokens: u32,
    output_tokens: u32,
    reasoning_tokens: Option<u32>,
    cache_read_tokens: Option<u32>,
    cache_creation_tokens: Option<u32>,
) -> serde_json::Value {
    let usage = gauss_core::message::Usage {
        input_tokens: input_tokens as u64,
        output_tokens: output_tokens as u64,
        reasoning_tokens: reasoning_tokens.map(|v| v as u64),
        cache_read_tokens: cache_read_tokens.map(|v| v as u64),
        cache_creation_tokens: cache_creation_tokens.map(|v| v as u64),
    };
    serde_json::to_value(gauss_core::cost::estimate_cost(&model, &usage))
        .unwrap_or_else(|_| json!({ "model": model, "currency": "USD", "total_cost_usd": 0.0 }))
}
