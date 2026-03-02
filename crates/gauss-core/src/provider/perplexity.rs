//! Perplexity provider — uses OpenAI-compatible API.

use crate::provider::ProviderConfig;
use crate::provider::openai::OpenAiProvider;
use crate::provider::openai_compatible::create_openai_compatible;

const PERPLEXITY_BASE_URL: &str = "https://api.perplexity.ai";

/// Perplexity provider — wraps OpenAI provider with Perplexity's base URL.
pub struct PerplexityProvider;

impl PerplexityProvider {
    /// Create a Perplexity provider.
    pub fn create(model: impl Into<String>, config: ProviderConfig) -> OpenAiProvider {
        create_openai_compatible(model, config, PERPLEXITY_BASE_URL)
    }
}
