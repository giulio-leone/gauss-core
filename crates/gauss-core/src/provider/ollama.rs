//! Ollama provider — uses OpenAI-compatible API.

use crate::provider::ProviderConfig;
use crate::provider::openai::OpenAiProvider;

const OLLAMA_BASE_URL: &str = "http://localhost:11434/v1";

/// Ollama provider — wraps OpenAI provider with Ollama's local base URL.
pub struct OllamaProvider;

impl OllamaProvider {
    /// Create an Ollama provider.
    /// Models: llama3.3, codellama, mistral, phi3, etc.
    /// Ollama does not require an API key — pass any string.
    pub fn create(model: impl Into<String>, mut config: ProviderConfig) -> OpenAiProvider {
        if config.base_url.is_none() {
            config.base_url = Some(OLLAMA_BASE_URL.to_string());
        }
        OpenAiProvider::new(model, config)
    }
}
