use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde_json::json;

use crate::error::{self, GaussError};
use crate::message::{Content, Message, Role, Usage};
use crate::provider::{
    FinishReason, GenerateOptions, GenerateResult, Provider, ProviderConfig, ReasoningEffort,
};
use crate::streaming::StreamEvent;
use crate::tool::Tool;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// OpenAI Chat Completions provider.
pub struct OpenAiProvider {
    config: ProviderConfig,
    model: String,
    client: Client,
}

impl OpenAiProvider {
    pub fn new(model: impl Into<String>, config: ProviderConfig) -> Self {
        let client = crate::provider::build_client(config.timeout_ms);

        Self {
            config,
            model: model.into(),
            client,
        }
    }

    fn base_url(&self) -> &str {
        self.config.base_url.as_deref().unwrap_or(DEFAULT_BASE_URL)
    }

    /// Returns true if this model requires `max_completion_tokens` instead of `max_tokens`.
    /// Newer OpenAI models (o-series reasoning, gpt-5.x) use the new parameter name.
    fn uses_max_completion_tokens(&self) -> bool {
        let m = self.model.as_str();
        m.starts_with("o1")
            || m.starts_with("o3")
            || m.starts_with("o4")
            || m.starts_with("gpt-5")
    }

    fn build_request_body(
        &self,
        messages: &[Message],
        tools: &[Tool],
        options: &GenerateOptions,
        stream: bool,
    ) -> serde_json::Value {
        let mut body = json!({
            "model": self.model,
            "messages": self.convert_messages(messages),
        });

        if stream {
            body["stream"] = json!(true);
            body["stream_options"] = json!({"include_usage": true});
        }

        if !tools.is_empty() {
            body["tools"] = json!(
                tools
                    .iter()
                    .map(|t| json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                    }))
                    .collect::<Vec<_>>()
            );

            if let Some(ref tc) = options.tool_choice {
                body["tool_choice"] = match tc {
                    crate::tool::ToolChoice::Auto => json!("auto"),
                    crate::tool::ToolChoice::None => json!("none"),
                    crate::tool::ToolChoice::Required => json!("required"),
                    crate::tool::ToolChoice::Specific { name } => json!({
                        "type": "function",
                        "function": {"name": name}
                    }),
                };
            }
        }

        if let Some(t) = options.temperature {
            body["temperature"] = json!(t);
        }
        if let Some(tp) = options.top_p {
            body["top_p"] = json!(tp);
        }
        if let Some(mt) = options.max_tokens {
            // Newer OpenAI models (o-series, gpt-5.x) require max_completion_tokens
            if self.uses_max_completion_tokens() {
                body["max_completion_tokens"] = json!(mt);
            } else {
                body["max_tokens"] = json!(mt);
            }
        }
        if let Some(fp) = options.frequency_penalty {
            body["frequency_penalty"] = json!(fp);
        }
        if let Some(pp) = options.presence_penalty {
            body["presence_penalty"] = json!(pp);
        }
        if let Some(seed) = options.seed {
            body["seed"] = json!(seed);
        }
        if let Some(ref stops) = options.stop_sequences {
            body["stop"] = json!(stops);
        }
        if let Some(ref re) = options.reasoning_effort {
            body["reasoning_effort"] = json!(match re {
                ReasoningEffort::Low => "low",
                ReasoningEffort::Medium => "medium",
                ReasoningEffort::High => "high",
            });
        }
        if let Some(ref schema) = options.output_schema {
            body["response_format"] = json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "output",
                    "schema": schema,
                    "strict": true,
                }
            });
        }

        body
    }

    fn convert_messages(&self, messages: &[Message]) -> Vec<serde_json::Value> {
        messages
            .iter()
            .map(|msg| {
                let role = match msg.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::Tool => "tool",
                };

                let mut m = json!({"role": role});

                match msg.role {
                    Role::Tool => {
                        if let Some(Content::ToolResult {
                            tool_call_id,
                            content,
                            ..
                        }) = msg.content.first()
                        {
                            m["tool_call_id"] = json!(tool_call_id);
                            m["content"] = json!(content.to_string());
                        }
                    }
                    Role::Assistant => {
                        let tool_calls: Vec<_> = msg
                            .content
                            .iter()
                            .filter_map(|c| match c {
                                Content::ToolCall {
                                    id,
                                    name,
                                    arguments,
                                } => Some(json!({
                                    "id": id,
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": arguments.to_string(),
                                    }
                                })),
                                _ => None,
                            })
                            .collect();

                        let text: String = msg
                            .content
                            .iter()
                            .filter_map(|c| match c {
                                Content::Text { text } => Some(text.as_str()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("");

                        if !text.is_empty() {
                            m["content"] = json!(text);
                        } else if !tool_calls.is_empty() {
                            m["content"] = serde_json::Value::Null;
                        }

                        if !tool_calls.is_empty() {
                            m["tool_calls"] = json!(tool_calls);
                        }
                    }
                    _ => {
                        // For system/user: serialize content parts
                        if msg.content.len() == 1 {
                            if let Some(Content::Text { text }) = msg.content.first() {
                                m["content"] = json!(text);
                            }
                        } else {
                            let parts: Vec<_> = msg
                                .content
                                .iter()
                                .filter_map(|c| match c {
                                    Content::Text { text } => {
                                        Some(json!({"type": "text", "text": text}))
                                    }
                                    Content::Image { url, base64, media_type } => {
                                        if let Some(url) = url {
                                            Some(json!({
                                                "type": "image_url",
                                                "image_url": {"url": url}
                                            }))
                                        } else if let Some(b64) = base64 {
                                            let mt = media_type.as_deref().unwrap_or("image/png");
                                            Some(json!({
                                                "type": "image_url",
                                                "image_url": {"url": format!("data:{mt};base64,{b64}")}
                                            }))
                                        } else {
                                            None
                                        }
                                    }
                                    _ => None,
                                })
                                .collect();
                            m["content"] = json!(parts);
                        }
                    }
                }

                m
            })
            .collect()
    }

    fn parse_response(&self, body: &serde_json::Value) -> error::Result<GenerateResult> {
        let choice = body["choices"]
            .as_array()
            .and_then(|c| c.first())
            .ok_or_else(|| GaussError::provider("openai", "No choices in response"))?;

        let msg = &choice["message"];
        let mut content = Vec::new();

        if let Some(text) = msg["content"].as_str()
            && !text.is_empty()
        {
            content.push(Content::Text {
                text: text.to_string(),
            });
        }

        if let Some(tool_calls) = msg["tool_calls"].as_array() {
            for tc in tool_calls {
                let id = tc["id"].as_str().unwrap_or("").to_string();
                let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
                let args_str = tc["function"]["arguments"].as_str().unwrap_or("{}");
                let arguments: serde_json::Value =
                    serde_json::from_str(args_str).unwrap_or(json!({}));
                content.push(Content::ToolCall {
                    id,
                    name,
                    arguments,
                });
            }
        }

        if let Some(reasoning) = msg.get("reasoning_content").and_then(|r| r.as_str()) {
            content.push(Content::Reasoning {
                text: reasoning.to_string(),
            });
        }

        let finish_reason = match choice["finish_reason"].as_str() {
            Some("stop") => FinishReason::Stop,
            Some("tool_calls") => FinishReason::ToolCalls,
            Some("length") => FinishReason::Length,
            Some("content_filter") => FinishReason::ContentFilter,
            Some(other) => FinishReason::Other(other.to_string()),
            None => FinishReason::Stop,
        };

        let usage_data = &body["usage"];
        let usage = Usage {
            input_tokens: usage_data["prompt_tokens"].as_u64().unwrap_or(0),
            output_tokens: usage_data["completion_tokens"].as_u64().unwrap_or(0),
            reasoning_tokens: usage_data["completion_tokens_details"]["reasoning_tokens"].as_u64(),
            cache_read_tokens: usage_data["prompt_tokens_details"]["cached_tokens"].as_u64(),
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
            provider_metadata: body
                .get("system_fingerprint")
                .cloned()
                .unwrap_or(json!(null)),
            thinking: None,
        })
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl Provider for OpenAiProvider {
    fn name(&self) -> &str {
        "openai"
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn generate(
        &self,
        messages: &[Message],
        tools: &[Tool],
        options: &GenerateOptions,
    ) -> error::Result<GenerateResult> {
        let body = self.build_request_body(messages, tools, options, false);
        let url = format!("{}/chat/completions", self.base_url());

        let mut req = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json");

        if let Some(ref org) = self.config.organization {
            req = req.header("OpenAI-Organization", org);
        }

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
                provider: "openai".to_string(),
                source: Some(Box::new(e)),
            })?;

        let status = resp.status();
        let resp_body: serde_json::Value = resp.json().await.map_err(|e| {
            GaussError::provider("openai", format!("Failed to parse response: {e}"))
        })?;

        if !status.is_success() {
            let error_msg = resp_body["error"]["message"]
                .as_str()
                .unwrap_or("Unknown error");

            if status.as_u16() == 429 {
                return Err(GaussError::RateLimited {
                    provider: "openai".to_string(),
                    retry_after_ms: None,
                });
            }

            if status.as_u16() == 401 {
                return Err(GaussError::Authentication {
                    provider: "openai".to_string(),
                });
            }

            return Err(GaussError::Provider {
                message: error_msg.to_string(),
                status: Some(status.as_u16()),
                provider: "openai".to_string(),
                source: None,
            });
        }

        self.parse_response(&resp_body)
    }

    async fn stream(
        &self,
        messages: &[Message],
        tools: &[Tool],
        options: &GenerateOptions,
    ) -> error::Result<crate::provider::BoxStream> {
        let body = self.build_request_body(messages, tools, options, true);
        let url = format!("{}/chat/completions", self.base_url());

        let mut req = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json");

        if let Some(ref org) = self.config.organization {
            req = req.header("OpenAI-Organization", org);
        }

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
                provider: "openai".to_string(),
                source: Some(Box::new(e)),
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body: serde_json::Value = resp.json().await.unwrap_or(json!({}));
            let msg = body["error"]["message"].as_str().unwrap_or("Unknown error");
            return Err(GaussError::Provider {
                message: msg.to_string(),
                status: Some(status.as_u16()),
                provider: "openai".to_string(),
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
                        if let Some(data) = line.strip_prefix("data: ") {
                            if data == "[DONE]" {
                                events.push(Ok(StreamEvent::Done));
                                continue;
                            }
                            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data) {
                                if let Some(choices) = parsed["choices"].as_array()
                                    && let Some(choice) = choices.first()
                                {
                                    let delta = &choice["delta"];

                                    if let Some(text) = delta["content"].as_str() {
                                        events.push(Ok(StreamEvent::TextDelta(text.to_string())));
                                    }

                                    if let Some(tcs) = delta["tool_calls"].as_array() {
                                        for tc in tcs {
                                            let index = tc["index"].as_u64().unwrap_or(0) as usize;
                                            events.push(Ok(StreamEvent::ToolCallDelta {
                                                index,
                                                id: tc["id"].as_str().map(String::from),
                                                name: tc["function"]["name"]
                                                    .as_str()
                                                    .map(String::from),
                                                arguments_delta: tc["function"]["arguments"]
                                                    .as_str()
                                                    .map(String::from),
                                            }));
                                        }
                                    }

                                    if let Some(reason) = choice["finish_reason"].as_str() {
                                        let fr = match reason {
                                            "stop" => FinishReason::Stop,
                                            "tool_calls" => FinishReason::ToolCalls,
                                            "length" => FinishReason::Length,
                                            "content_filter" => FinishReason::ContentFilter,
                                            other => FinishReason::Other(other.to_string()),
                                        };
                                        events.push(Ok(StreamEvent::FinishReason(fr)));
                                    }
                                }

                                if let Some(usage) = parsed.get("usage")
                                    && !usage.is_null()
                                {
                                    events.push(Ok(StreamEvent::Usage(Usage {
                                        input_tokens: usage["prompt_tokens"].as_u64().unwrap_or(0),
                                        output_tokens: usage["completion_tokens"]
                                            .as_u64()
                                            .unwrap_or(0),
                                        reasoning_tokens: None,
                                        cache_read_tokens: None,
                                        cache_creation_tokens: None,
                                    })));
                                }
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
