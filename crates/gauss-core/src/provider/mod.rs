use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error;
use crate::message::{Message, Usage};
use crate::streaming::StreamEvent;
use crate::tool::{Tool, ToolChoice};

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

/// Core trait for AI model providers.
#[async_trait]
pub trait Provider: Send + Sync {
    /// Provider name (e.g., "openai", "anthropic", "google").
    fn name(&self) -> &str;

    /// Model identifier (e.g., "gpt-5.2", "claude-sonnet-4-20250514").
    fn model(&self) -> &str;

    /// Generate a complete response.
    async fn generate(
        &self,
        messages: &[Message],
        tools: &[Tool],
        options: &GenerateOptions,
    ) -> error::Result<GenerateResult>;

    /// Generate a streaming response.
    async fn stream(
        &self,
        messages: &[Message],
        tools: &[Tool],
        options: &GenerateOptions,
    ) -> error::Result<Box<dyn futures::Stream<Item = error::Result<StreamEvent>> + Send + Unpin>>;
}

pub mod anthropic;
pub mod google;
pub mod openai;
pub mod retry;
