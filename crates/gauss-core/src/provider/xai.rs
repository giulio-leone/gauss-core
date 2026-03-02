//! xAI provider — uses OpenAI-compatible API.

use crate::provider::ProviderConfig;
use crate::provider::openai::OpenAiProvider;
use crate::provider::openai_compatible::create_openai_compatible;

const XAI_BASE_URL: &str = "https://api.x.ai/v1";

/// xAI provider — wraps OpenAI provider with xAI's base URL.
pub struct XaiProvider;

impl XaiProvider {
    /// Create an xAI provider.
    pub fn create(model: impl Into<String>, config: ProviderConfig) -> OpenAiProvider {
        create_openai_compatible(model, config, XAI_BASE_URL)
    }
}
