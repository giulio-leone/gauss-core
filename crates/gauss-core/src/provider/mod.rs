use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error;
use crate::message::{Message, Usage};
use crate::streaming::StreamEvent;
use crate::tool::{Tool, ToolChoice};

/// Platform-aware boxed stream type.
#[cfg(not(target_arch = "wasm32"))]
pub type BoxStream = Box<dyn futures::Stream<Item = error::Result<StreamEvent>> + Send + Unpin>;
#[cfg(target_arch = "wasm32")]
pub type BoxStream = Box<dyn futures::Stream<Item = error::Result<StreamEvent>> + Unpin>;

/// Reasoning effort level.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}

/// Options for text generation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerateOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<serde_json::Value>,
    /// Anthropic extended thinking: budget in tokens for internal reasoning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<u32>,
    /// Enable prompt caching (Anthropic). Auto-annotates system messages and tools with cache_control.
    #[serde(default)]
    pub cache_control: bool,
}

/// Provider configuration.
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub api_key: String,
    pub base_url: Option<String>,
    pub headers: Vec<(String, String)>,
    pub timeout_ms: Option<u64>,
    pub max_retries: Option<u32>,
    pub organization: Option<String>,
}

impl ProviderConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: None,
            headers: Vec::new(),
            timeout_ms: Some(60_000),
            max_retries: Some(3),
            organization: None,
        }
    }

    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    pub fn timeout(mut self, ms: u64) -> Self {
        self.timeout_ms = Some(ms);
        self
    }
}

/// Result from a non-streaming generation.
#[derive(Debug, Clone)]
pub struct GenerateResult {
    pub message: Message,
    pub usage: Usage,
    pub finish_reason: FinishReason,
    pub provider_metadata: serde_json::Value,
    /// Anthropic extended thinking output (if enabled).
    pub thinking: Option<String>,
    /// Citations from document-aware responses (Anthropic).
    pub citations: Vec<crate::message::Citation>,
}

impl GenerateResult {
    pub fn text(&self) -> Option<&str> {
        self.message.text()
    }

    pub fn tool_calls(&self) -> Vec<(&str, &str, &serde_json::Value)> {
        self.message.tool_calls()
    }
}

/// Reason the model stopped generating.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    ToolCalls,
    Length,
    ContentFilter,
    Error,
    Other(String),
}

/// Capabilities supported by a provider/model combination.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProviderCapabilities {
    pub streaming: bool,
    pub tool_use: bool,
    pub vision: bool,
    pub audio: bool,
    pub extended_thinking: bool,
    pub citations: bool,
    pub cache_control: bool,
    pub structured_output: bool,
    pub reasoning_effort: bool,
    pub image_generation: bool,
    pub grounding: bool,
    pub code_execution: bool,
    pub web_search: bool,
}

/// Core trait for AI model providers.
#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> &str;
    fn model(&self) -> &str;

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            streaming: true,
            tool_use: true,
            ..Default::default()
        }
    }

    async fn generate(
        &self,
        messages: &[Message],
        tools: &[Tool],
        options: &GenerateOptions,
    ) -> error::Result<GenerateResult>;

    async fn stream(
        &self,
        messages: &[Message],
        tools: &[Tool],
        options: &GenerateOptions,
    ) -> error::Result<BoxStream>;
}

/// Core trait for AI model providers (WASM — no Send + Sync).
#[cfg(target_arch = "wasm32")]
#[async_trait(?Send)]
pub trait Provider {
    fn name(&self) -> &str;
    fn model(&self) -> &str;

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            streaming: true,
            tool_use: true,
            ..Default::default()
        }
    }

    async fn generate(
        &self,
        messages: &[Message],
        tools: &[Tool],
        options: &GenerateOptions,
    ) -> error::Result<GenerateResult>;

    async fn stream(
        &self,
        messages: &[Message],
        tools: &[Tool],
        options: &GenerateOptions,
    ) -> error::Result<BoxStream>;
}

/// Build an HTTP client with platform-appropriate settings.
pub fn build_client(timeout_ms: Option<u64>) -> reqwest::Client {
    let builder = reqwest::Client::builder();
    #[cfg(not(target_arch = "wasm32"))]
    let builder = builder.timeout(std::time::Duration::from_millis(
        timeout_ms.unwrap_or(60_000),
    ));
    let _ = timeout_ms; // suppress unused warning on wasm
    builder.build().expect("Failed to build HTTP client")
}

pub mod anthropic;
pub mod deepseek;
pub mod google;
pub mod groq;
pub mod ollama;
pub mod openai;
pub mod retry;
