//! Groq provider — uses OpenAI-compatible API.

use crate::provider::ProviderConfig;
use crate::provider::openai::OpenAiProvider;
use crate::provider::openai_compatible::create_openai_compatible;

const GROQ_BASE_URL: &str = "https://api.groq.com/openai/v1";

/// Groq provider — wraps OpenAI provider with Groq's base URL.
pub struct GroqProvider;

impl GroqProvider {
    /// Create a Groq provider.
    /// Models: llama-3.3-70b-versatile, mixtral-8x7b-32768, gemma2-9b-it, etc.
    pub fn create(model: impl Into<String>, config: ProviderConfig) -> OpenAiProvider {
        create_openai_compatible(model, config, GROQ_BASE_URL)
    }
}
