//! Fireworks provider — uses OpenAI-compatible API.

use crate::provider::ProviderConfig;
use crate::provider::openai::OpenAiProvider;
use crate::provider::openai_compatible::create_openai_compatible;

const FIREWORKS_BASE_URL: &str = "https://api.fireworks.ai/inference/v1";

/// Fireworks provider — wraps OpenAI provider with Fireworks' base URL.
pub struct FireworksProvider;

impl FireworksProvider {
    /// Create a Fireworks provider.
    pub fn create(model: impl Into<String>, config: ProviderConfig) -> OpenAiProvider {
        create_openai_compatible(model, config, FIREWORKS_BASE_URL)
    }
}
