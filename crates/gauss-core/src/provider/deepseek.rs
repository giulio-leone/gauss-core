//! DeepSeek provider — uses OpenAI-compatible API.

use crate::provider::ProviderConfig;
use crate::provider::openai::OpenAiProvider;

const DEEPSEEK_BASE_URL: &str = "https://api.deepseek.com/v1";

/// DeepSeek provider — wraps OpenAI provider with DeepSeek's base URL.
pub struct DeepSeekProvider;

impl DeepSeekProvider {
    /// Create a DeepSeek provider.
    /// Models: deepseek-chat, deepseek-coder, deepseek-reasoner
    pub fn create(model: impl Into<String>, mut config: ProviderConfig) -> OpenAiProvider {
        config.base_url = Some(DEEPSEEK_BASE_URL.to_string());
        OpenAiProvider::new(model, config)
    }
}
