#[macro_use]
extern crate napi_derive;

use gauss_core::provider::ProviderConfig;
use gauss_core::provider::openai::OpenAiProvider;
use napi::bindgen_prelude::*;
use std::sync::Arc;

/// Gauss Core version.
#[napi]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Create an OpenAI provider.
#[napi]
pub fn create_openai_provider(model: String, api_key: String) -> Result<()> {
    let _provider = OpenAiProvider::new(model, ProviderConfig::new(api_key));
    Ok(())
}
