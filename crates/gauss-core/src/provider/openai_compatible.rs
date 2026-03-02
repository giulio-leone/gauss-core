//! Shared helper for providers exposing an OpenAI-compatible API.

use crate::provider::ProviderConfig;
use crate::provider::openai::OpenAiProvider;

pub fn create_openai_compatible(
    model: impl Into<String>,
    mut config: ProviderConfig,
    default_base_url: &str,
) -> OpenAiProvider {
    if config.base_url.is_none() {
        config.base_url = Some(default_base_url.to_string());
    }
    OpenAiProvider::new(model, config)
}
