//! OpenRouter provider — uses OpenAI-compatible API.

use crate::provider::ProviderConfig;
use crate::provider::openai::OpenAiProvider;
use crate::provider::openai_compatible::create_openai_compatible;

const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

/// OpenRouter provider — wraps OpenAI provider with OpenRouter's base URL.
pub struct OpenRouterProvider;

impl OpenRouterProvider {
    /// Create an OpenRouter provider.
    pub fn create(model: impl Into<String>, config: ProviderConfig) -> OpenAiProvider {
        create_openai_compatible(model, config, OPENROUTER_BASE_URL)
    }
}
