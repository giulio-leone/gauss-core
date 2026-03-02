//! Mistral provider — uses OpenAI-compatible API.

use crate::provider::ProviderConfig;
use crate::provider::openai::OpenAiProvider;
use crate::provider::openai_compatible::create_openai_compatible;

const MISTRAL_BASE_URL: &str = "https://api.mistral.ai/v1";

/// Mistral provider — wraps OpenAI provider with Mistral's base URL.
pub struct MistralProvider;

impl MistralProvider {
    /// Create a Mistral provider.
    pub fn create(model: impl Into<String>, config: ProviderConfig) -> OpenAiProvider {
        create_openai_compatible(model, config, MISTRAL_BASE_URL)
    }
}
