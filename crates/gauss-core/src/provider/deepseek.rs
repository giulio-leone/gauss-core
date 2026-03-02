//! DeepSeek provider — uses OpenAI-compatible API.

use crate::provider::ProviderConfig;
use crate::provider::openai::OpenAiProvider;
use crate::provider::openai_compatible::create_openai_compatible;

const DEEPSEEK_BASE_URL: &str = "https://api.deepseek.com/v1";

/// DeepSeek provider — wraps OpenAI provider with DeepSeek's base URL.
pub struct DeepSeekProvider;

impl DeepSeekProvider {
    /// Create a DeepSeek provider.
    /// Models: deepseek-chat, deepseek-coder, deepseek-reasoner
    pub fn create(model: impl Into<String>, config: ProviderConfig) -> OpenAiProvider {
        create_openai_compatible(model, config, DEEPSEEK_BASE_URL)
    }
}
