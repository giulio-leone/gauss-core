//! Together AI provider — uses OpenAI-compatible API.

use crate::provider::ProviderConfig;
use crate::provider::openai::OpenAiProvider;
use crate::provider::openai_compatible::create_openai_compatible;

const TOGETHER_BASE_URL: &str = "https://api.together.xyz/v1";

/// Together provider — wraps OpenAI provider with Together's base URL.
pub struct TogetherProvider;

impl TogetherProvider {
    /// Create a Together provider.
    pub fn create(model: impl Into<String>, config: ProviderConfig) -> OpenAiProvider {
        create_openai_compatible(model, config, TOGETHER_BASE_URL)
    }
}
