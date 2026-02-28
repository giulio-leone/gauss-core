//! Groq provider — uses OpenAI-compatible API.

use crate::provider::ProviderConfig;
use crate::provider::openai::OpenAiProvider;

const GROQ_BASE_URL: &str = "https://api.groq.com/openai/v1";

/// Groq provider — wraps OpenAI provider with Groq's base URL.
pub struct GroqProvider;

impl GroqProvider {
    /// Create a Groq provider.
    /// Models: llama-3.3-70b-versatile, mixtral-8x7b-32768, gemma2-9b-it, etc.
    pub fn create(model: impl Into<String>, mut config: ProviderConfig) -> OpenAiProvider {
        config.base_url = Some(GROQ_BASE_URL.to_string());
        OpenAiProvider::new(model, config)
    }
}
