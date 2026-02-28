use async_trait::async_trait;
use futures::StreamExt;
use serde_json::json;
use tracing::debug;

use crate::error::GaussError;
use crate::message::{Content, Message, Role, Usage};
use crate::provider::{FinishReason, GenerateOptions, GenerateResult, Provider, ProviderConfig};
use crate::streaming::StreamEvent;
use crate::tool::Tool;

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";
const API_VERSION: &str = "2023-06-01";

/// Anthropic Messages API provider.
pub struct AnthropicProvider {
    model: String,
    config: ProviderConfig,
    client: reqwest::Client,
}

impl AnthropicProvider {
    pub fn new(model: impl Into<String>, config: ProviderConfig) -> Self {
        let client = crate::provider::build_client(config.timeout_ms);

        Self {
            model: model.into(),
            config,
            client,
        }
    }

    fn base_url(&self) -> &str {
        self.config.base_url.as_deref().unwrap_or(DEFAULT_BASE_URL)
    }

    fn build_request_body(
        &self,
        messages: &[Message],
        tools: &[Tool],
        options: &GenerateOptions,
    ) -> (Option<String>, serde_json::Value) {
        // Extract system message separately (Anthropic uses top-level `system`)
        let mut system_text: Option<String> = None;
        let api_messages: Vec<serde_json::Value> = messages
            .iter()
            .filter_map(|m| {
                match m.role {
                    Role::System => {
                        if let Some(text) = m.text() {
                            system_text = Some(text.to_string());
                        }
                        None // system messages go top-level
                    }
                    Role::User => Some(json!({
                        "role": "user",
                        "content": self.convert_content(&m.content)
                    })),
                    Role::Assistant => {
                        let mut msg = json!({"role": "assistant"});
                        let content = self.convert_assistant_content(&m.content);
                        if !content.is_empty() {
                            msg["content"] = json!(content);
                        }
                        Some(msg)
                    }
                    Role::Tool => {
                        // Anthropic expects tool results as user messages with tool_result blocks
                        let results: Vec<serde_json::Value> = m
                            .content
                            .iter()
                            .filter_map(|c| {
                                if let Content::ToolResult {
                                    tool_call_id,
                                    content,
                                    is_error,
                                } = c
                                {
                                    Some(json!({
                                        "type": "tool_result",
                                        "tool_use_id": tool_call_id,
                                        "content": content.to_string(),
                                        "is_error": is_error
                                    }))
                                } else {
                                    None
                                }
                            })
                            .collect();
                        if results.is_empty() {
                            None
                        } else {
                            Some(json!({"role": "user", "content": results}))
                        }
                    }
                }
            })
            .collect();

        let mut body = json!({
            "model": self.model,
            "messages": api_messages,
            "max_tokens": options.max_tokens.unwrap_or(4096),
        });

        if !tools.is_empty() {
            body["tools"] = json!(
                tools
                    .iter()
                    .map(|t| json!({
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.parameters,
                    }))
                    .collect::<Vec<_>>()
            );

            if let Some(ref tc) = options.tool_choice {
                body["tool_choice"] = match tc {
                    crate::tool::ToolChoice::Auto => json!({"type": "auto"}),
                    crate::tool::ToolChoice::None => json!({"type": "none"}), // not officially supported but maps
                    crate::tool::ToolChoice::Required => json!({"type": "any"}),
                    crate::tool::ToolChoice::Specific { name } => {
                        json!({"type": "tool", "name": name})
                    }
                };
            }
        }

        if let Some(t) = options.temperature {
            body["temperature"] = json!(t);
        }
        if let Some(tp) = options.top_p {
            body["top_p"] = json!(tp);
        }
        if let Some(tk) = options.top_k {
            body["top_k"] = json!(tk);
        }
        if let Some(ref stops) = options.stop_sequences {
            body["stop_sequences"] = json!(stops);
        }
        if let Some(ref schema) = options.output_schema {
            // Anthropic uses prefill + tool for structured output — simplified here
            body["metadata"] = json!({"output_schema": schema});
        }

        (system_text, body)
    }

    fn convert_content(&self, content: &[Content]) -> serde_json::Value {
        let parts: Vec<serde_json::Value> = content
            .iter()
            .filter_map(|c| match c {
                Content::Text { text } => Some(json!({"type": "text", "text": text})),
                Content::Image {
                    base64, media_type, ..
                } => Some(json!({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64
                    }
                })),
                _ => None,
            })
            .collect();

        if parts.len() == 1
            && let Some(text) = content.first()
            && let Content::Text { text } = text
        {
            return json!(text);
        }
        json!(parts)
    }

    fn convert_assistant_content(&self, content: &[Content]) -> Vec<serde_json::Value> {
        content
            .iter()
            .filter_map(|c| match c {
                Content::Text { text } => Some(json!({"type": "text", "text": text})),
                Content::ToolCall {
                    id,
                    name,
                    arguments: args,
                } => Some(json!({
                    "type": "tool_use",
                    "id": id,
                    "name": name,
                    "input": args
                })),
                _ => None,
            })
            .collect()
    }

    fn parse_response(&self, body: &serde_json::Value) -> crate::error::Result<GenerateResult> {
        let mut content = Vec::new();

        if let Some(blocks) = body["content"].as_array() {
            for block in blocks {
                match block["type"].as_str() {
                    Some("text") => {
                        if let Some(text) = block["text"].as_str() {
                            content.push(Content::Text {
                                text: text.to_string(),
                            });
                        }
                    }
                    Some("tool_use") => {
                        let id = block["id"].as_str().unwrap_or("").to_string();
                        let name = block["name"].as_str().unwrap_or("").to_string();
                        let args = block["input"].clone();
                        content.push(Content::ToolCall {
                            id,
                            name,
                            arguments: args,
                        });
                    }
                    _ => {}
                }
            }
        }

        let stop_reason = body["stop_reason"].as_str().unwrap_or("end_turn");
        let finish_reason = match stop_reason {
            "end_turn" | "stop" => FinishReason::Stop,
            "tool_use" => FinishReason::ToolCalls,
            "max_tokens" => FinishReason::Length,
            other => FinishReason::Other(other.to_string()),
        };

        let usage = Usage {
            input_tokens: body["usage"]["input_tokens"].as_u64().unwrap_or(0),
            output_tokens: body["usage"]["output_tokens"].as_u64().unwrap_or(0),
            reasoning_tokens: body["usage"]["cache_creation_input_tokens"].as_u64(),
            cache_read_tokens: body["usage"]["cache_read_input_tokens"].as_u64(),
            cache_creation_tokens: None,
        };

        Ok(GenerateResult {
            message: Message {
                role: Role::Assistant,
                content,
                name: None,
            },
            usage,
            finish_reason,
            provider_metadata: body.get("id").cloned().unwrap_or(json!(null)),
        })
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn generate(
        &self,
        messages: &[Message],
        tools: &[Tool],
        options: &GenerateOptions,
    ) -> crate::error::Result<GenerateResult> {
        let url = format!("{}/messages", self.base_url());
        let (system, mut body) = self.build_request_body(messages, tools, options);

        if let Some(sys) = system {
            body["system"] = json!(sys);
        }

        debug!(model = %self.model, url = %url, "Anthropic generate");

        let mut req = self
            .client
            .post(&url)
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json");

        for (k, v) in &self.config.headers {
            req = req.header(k, v);
        }

        let resp = req
            .json(&body)
            .send()
            .await
            .map_err(|e| GaussError::Provider {
                message: e.to_string(),
                status: e.status().map(|s| s.as_u16()),
                provider: "anthropic".to_string(),
                source: Some(Box::new(e)),
            })?;

        let status = resp.status();
        let resp_body: serde_json::Value = resp.json().await.map_err(|e| {
            GaussError::provider("anthropic", format!("Failed to parse response: {e}"))
        })?;

        if !status.is_success() {
            let msg = resp_body["error"]["message"]
                .as_str()
                .unwrap_or("Unknown error");

            return match status.as_u16() {
                429 => Err(GaussError::rate_limited("anthropic", msg)),
                401 => Err(GaussError::authentication("anthropic", msg)),
                _ => Err(GaussError::Provider {
                    message: msg.to_string(),
                    status: Some(status.as_u16()),
                    provider: "anthropic".to_string(),
                    source: None,
                }),
            };
        }

        self.parse_response(&resp_body)
    }

    async fn stream(
        &self,
        messages: &[Message],
        tools: &[Tool],
        options: &GenerateOptions,
    ) -> crate::error::Result<
        crate::provider::BoxStream,
    > {
        let url = format!("{}/messages", self.base_url());
        let (system, mut body) = self.build_request_body(messages, tools, options);

        if let Some(sys) = system {
            body["system"] = json!(sys);
        }
        body["stream"] = json!(true);

        debug!(model = %self.model, url = %url, "Anthropic stream");

        let mut req = self
            .client
            .post(&url)
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json");

        for (k, v) in &self.config.headers {
            req = req.header(k, v);
        }

        let resp = req
            .json(&body)
            .send()
            .await
            .map_err(|e| GaussError::Provider {
                message: e.to_string(),
                status: e.status().map(|s| s.as_u16()),
                provider: "anthropic".to_string(),
                source: Some(Box::new(e)),
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body: serde_json::Value = resp.json().await.unwrap_or(json!({}));
            let msg = body["error"]["message"].as_str().unwrap_or("Unknown error");
            return Err(GaussError::Provider {
                message: msg.to_string(),
                status: Some(status.as_u16()),
                provider: "anthropic".to_string(),
                source: None,
            });
        }

        let byte_stream = resp.bytes_stream();
        let stream = byte_stream
            .map(move |chunk| match chunk {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    let mut events = Vec::new();

                    for line in text.lines() {
                        let line = line.trim();
                        if let Some(data) = line.strip_prefix("data: ")
                            && let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data)
                        {
                            match parsed["type"].as_str() {
                                Some("content_block_delta") => {
                                    let delta = &parsed["delta"];
                                    match delta["type"].as_str() {
                                        Some("text_delta") => {
                                            if let Some(text) = delta["text"].as_str() {
                                                events.push(Ok(StreamEvent::TextDelta(
                                                    text.to_string(),
                                                )));
                                            }
                                        }
                                        Some("input_json_delta") => {
                                            if let Some(json) = delta["partial_json"].as_str() {
                                                events.push(Ok(StreamEvent::ToolCallDelta {
                                                    index: parsed["index"].as_u64().unwrap_or(0)
                                                        as usize,
                                                    id: None,
                                                    name: None,
                                                    arguments_delta: Some(json.to_string()),
                                                }));
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                                Some("content_block_start") => {
                                    let cb = &parsed["content_block"];
                                    if cb["type"].as_str() == Some("tool_use") {
                                        events.push(Ok(StreamEvent::ToolCallDelta {
                                            index: parsed["index"].as_u64().unwrap_or(0) as usize,
                                            id: cb["id"].as_str().map(String::from),
                                            name: cb["name"].as_str().map(String::from),
                                            arguments_delta: None,
                                        }));
                                    }
                                }
                                Some("message_delta") => {
                                    let delta = &parsed["delta"];
                                    if let Some(reason) = delta["stop_reason"].as_str() {
                                        let fr = match reason {
                                            "end_turn" => FinishReason::Stop,
                                            "tool_use" => FinishReason::ToolCalls,
                                            "max_tokens" => FinishReason::Length,
                                            other => FinishReason::Other(other.to_string()),
                                        };
                                        events.push(Ok(StreamEvent::FinishReason(fr)));
                                    }
                                    if let Some(usage) = parsed.get("usage")
                                        && !usage.is_null()
                                    {
                                        events.push(Ok(StreamEvent::Usage(Usage {
                                            input_tokens: 0,
                                            output_tokens: usage["output_tokens"]
                                                .as_u64()
                                                .unwrap_or(0),
                                            reasoning_tokens: None,
                                            cache_read_tokens: None,
                                            cache_creation_tokens: None,
                                        })));
                                    }
                                }
                                Some("message_start") => {
                                    if let Some(msg) = parsed.get("message")
                                        && let Some(usage) = msg.get("usage")
                                        && !usage.is_null()
                                    {
                                        events.push(Ok(StreamEvent::Usage(Usage {
                                            input_tokens: usage["input_tokens"]
                                                .as_u64()
                                                .unwrap_or(0),
                                            output_tokens: usage["output_tokens"]
                                                .as_u64()
                                                .unwrap_or(0),
                                            reasoning_tokens: None,
                                            cache_read_tokens: None,
                                            cache_creation_tokens: None,
                                        })));
                                    }
                                }
                                Some("message_stop") => {
                                    events.push(Ok(StreamEvent::Done));
                                }
                                _ => {}
                            }
                        }
                    }

                    futures::stream::iter(events)
                }
                Err(e) => futures::stream::iter(vec![Err(GaussError::Stream {
                    message: e.to_string(),
                    source: Some(Box::new(e)),
                })]),
            })
            .flatten();

        Ok(Box::new(stream))
    }
}
