use async_trait::async_trait;
use futures::StreamExt;
use serde_json::json;
use tracing::debug;

use crate::error::GaussError;
use crate::message::{Content, Message, Role, Usage};
use crate::provider::{FinishReason, GenerateOptions, GenerateResult, Provider, ProviderConfig};
use crate::streaming::StreamEvent;
use crate::tool::Tool;

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

/// Google Generative AI (Gemini) provider.
pub struct GoogleProvider {
    model: String,
    config: ProviderConfig,
    client: reqwest::Client,
}

impl GoogleProvider {
    pub fn new(model: impl Into<String>, config: ProviderConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(
                config.timeout_ms.unwrap_or(60_000),
            ))
            .build()
            .expect("Failed to create HTTP client");

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
    ) -> serde_json::Value {
        let mut system_instruction: Option<serde_json::Value> = None;

        let contents: Vec<serde_json::Value> = messages
            .iter()
            .filter_map(|m| match m.role {
                Role::System => {
                    if let Some(text) = m.text() {
                        system_instruction = Some(json!({
                            "parts": [{"text": text}]
                        }));
                    }
                    None
                }
                Role::User => Some(json!({
                    "role": "user",
                    "parts": self.convert_parts(&m.content)
                })),
                Role::Assistant => Some(json!({
                    "role": "model",
                    "parts": self.convert_model_parts(&m.content)
                })),
                Role::Tool => {
                    let parts: Vec<serde_json::Value> = m
                        .content
                        .iter()
                        .filter_map(|c| {
                            if let Content::ToolResult {
                                tool_call_id,
                                content,
                                ..
                            } = c
                            {
                                Some(json!({
                                    "functionResponse": {
                                        "name": tool_call_id,
                                        "response": content
                                    }
                                }))
                            } else {
                                None
                            }
                        })
                        .collect();
                    if parts.is_empty() {
                        None
                    } else {
                        Some(json!({"role": "function", "parts": parts}))
                    }
                }
            })
            .collect();

        let mut body = json!({"contents": contents});

        if let Some(si) = system_instruction {
            body["systemInstruction"] = si;
        }

        // Generation config
        let mut gen_config = json!({});
        if let Some(t) = options.temperature {
            gen_config["temperature"] = json!(t);
        }
        if let Some(tp) = options.top_p {
            gen_config["topP"] = json!(tp);
        }
        if let Some(tk) = options.top_k {
            gen_config["topK"] = json!(tk);
        }
        if let Some(mt) = options.max_tokens {
            gen_config["maxOutputTokens"] = json!(mt);
        }
        if let Some(ref stops) = options.stop_sequences {
            gen_config["stopSequences"] = json!(stops);
        }
        if let Some(ref schema) = options.output_schema {
            gen_config["responseMimeType"] = json!("application/json");
            gen_config["responseSchema"] = schema.clone();
        }
        if gen_config.as_object().is_some_and(|o| !o.is_empty()) {
            body["generationConfig"] = gen_config;
        }

        // Tools
        if !tools.is_empty() {
            let function_declarations: Vec<serde_json::Value> = tools
                .iter()
                .map(|t| {
                    json!({
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    })
                })
                .collect();
            body["tools"] = json!([{"functionDeclarations": function_declarations}]);
        }

        body
    }

    fn convert_parts(&self, content: &[Content]) -> Vec<serde_json::Value> {
        content
            .iter()
            .filter_map(|c| match c {
                Content::Text { text } => Some(json!({"text": text})),
                Content::Image {
                    base64, media_type, ..
                } => Some(json!({
                    "inlineData": {
                        "mimeType": media_type,
                        "data": base64
                    }
                })),
                _ => None,
            })
            .collect()
    }

    fn convert_model_parts(&self, content: &[Content]) -> Vec<serde_json::Value> {
        content
            .iter()
            .filter_map(|c| match c {
                Content::Text { text } => Some(json!({"text": text})),
                Content::ToolCall {
                    name, arguments, ..
                } => Some(json!({
                    "functionCall": {
                        "name": name,
                        "args": arguments
                    }
                })),
                _ => None,
            })
            .collect()
    }

    fn parse_response(&self, body: &serde_json::Value) -> crate::error::Result<GenerateResult> {
        let candidate = body["candidates"]
            .as_array()
            .and_then(|c| c.first())
            .ok_or_else(|| GaussError::provider("google", "No candidates in response"))?;

        let mut content = Vec::new();

        if let Some(parts) = candidate["content"]["parts"].as_array() {
            for part in parts {
                if let Some(text) = part["text"].as_str() {
                    content.push(Content::Text {
                        text: text.to_string(),
                    });
                }
                if let Some(fc) = part.get("functionCall") {
                    let name = fc["name"].as_str().unwrap_or("").to_string();
                    let arguments = fc["args"].clone();
                    content.push(Content::ToolCall {
                        id: name.clone(),
                        name,
                        arguments,
                    });
                }
            }
        }

        let finish_reason = match candidate["finishReason"].as_str() {
            Some("STOP") | None => FinishReason::Stop,
            Some("MAX_TOKENS") => FinishReason::Length,
            Some("SAFETY") => FinishReason::ContentFilter,
            Some(other) => FinishReason::Other(other.to_string()),
        };

        let usage_meta = &body["usageMetadata"];
        let usage = Usage {
            input_tokens: usage_meta["promptTokenCount"].as_u64().unwrap_or(0),
            output_tokens: usage_meta["candidatesTokenCount"].as_u64().unwrap_or(0),
            reasoning_tokens: usage_meta["thoughtsTokenCount"].as_u64(),
            cache_read_tokens: usage_meta["cachedContentTokenCount"].as_u64(),
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
            provider_metadata: body.get("modelVersion").cloned().unwrap_or(json!(null)),
        })
    }
}

#[async_trait]
impl Provider for GoogleProvider {
    fn name(&self) -> &str {
        "google"
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
        let url = format!(
            "{}/models/{}:generateContent?key={}",
            self.base_url(),
            self.model,
            self.config.api_key
        );
        let body = self.build_request_body(messages, tools, options);

        debug!(model = %self.model, "Google generate");

        let mut req = self
            .client
            .post(&url)
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
                provider: "google".to_string(),
                source: Some(Box::new(e)),
            })?;

        let status = resp.status();
        let resp_body: serde_json::Value = resp.json().await.map_err(|e| {
            GaussError::provider("google", format!("Failed to parse response: {e}"))
        })?;

        if !status.is_success() {
            let msg = resp_body["error"]["message"]
                .as_str()
                .unwrap_or("Unknown error");

            return match status.as_u16() {
                429 => Err(GaussError::rate_limited("google", msg)),
                401 | 403 => Err(GaussError::authentication("google", msg)),
                _ => Err(GaussError::Provider {
                    message: msg.to_string(),
                    status: Some(status.as_u16()),
                    provider: "google".to_string(),
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
        Box<dyn futures::Stream<Item = crate::error::Result<StreamEvent>> + Send + Unpin>,
    > {
        let url = format!(
            "{}/models/{}:streamGenerateContent?alt=sse&key={}",
            self.base_url(),
            self.model,
            self.config.api_key
        );
        let body = self.build_request_body(messages, tools, options);

        debug!(model = %self.model, "Google stream");

        let mut req = self
            .client
            .post(&url)
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
                provider: "google".to_string(),
                source: Some(Box::new(e)),
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body: serde_json::Value = resp.json().await.unwrap_or(json!({}));
            let msg = body["error"]["message"].as_str().unwrap_or("Unknown error");
            return Err(GaussError::Provider {
                message: msg.to_string(),
                status: Some(status.as_u16()),
                provider: "google".to_string(),
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
                            if let Some(candidates) = parsed["candidates"].as_array()
                                && let Some(candidate) = candidates.first()
                                && let Some(parts) = candidate["content"]["parts"].as_array()
                            {
                                for part in parts {
                                    if let Some(text) = part["text"].as_str() {
                                        events.push(Ok(StreamEvent::TextDelta(text.to_string())));
                                    }
                                    if let Some(fc) = part.get("functionCall") {
                                        events.push(Ok(StreamEvent::ToolCallDelta {
                                            index: 0,
                                            id: fc["name"].as_str().map(String::from),
                                            name: fc["name"].as_str().map(String::from),
                                            arguments_delta: Some(fc["args"].to_string()),
                                        }));
                                    }
                                }

                                if let Some(reason) = candidate["finishReason"].as_str() {
                                    let fr = match reason {
                                        "STOP" => FinishReason::Stop,
                                        "MAX_TOKENS" => FinishReason::Length,
                                        "SAFETY" => FinishReason::ContentFilter,
                                        other => FinishReason::Other(other.to_string()),
                                    };
                                    events.push(Ok(StreamEvent::FinishReason(fr)));
                                }
                            }

                            if let Some(usage) = parsed.get("usageMetadata")
                                && !usage.is_null()
                            {
                                events.push(Ok(StreamEvent::Usage(Usage {
                                    input_tokens: usage["promptTokenCount"].as_u64().unwrap_or(0),
                                    output_tokens: usage["candidatesTokenCount"]
                                        .as_u64()
                                        .unwrap_or(0),
                                    reasoning_tokens: None,
                                    cache_read_tokens: None,
                                    cache_creation_tokens: None,
                                })));
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
