use async_trait::async_trait;
use futures::StreamExt;
use serde_json::json;
use tracing::debug;

use crate::error::GaussError;
use crate::message::{Content, Message, Role, Usage};
use crate::provider::{FinishReason, GenerateOptions, GenerateResult, Provider, ProviderCapabilities, ProviderConfig};
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

        // Tools (function declarations + provider features)
        let mut tools_array: Vec<serde_json::Value> = Vec::new();

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
            tools_array.push(json!({"functionDeclarations": function_declarations}));
        }

        // Google Search grounding
        if options.grounding {
            tools_array.push(json!({"google_search": {}}));
        }

        // Native code execution (Gemini code interpreter)
        if options.native_code_execution {
            tools_array.push(json!({"codeExecution": {}}));
        }

        if !tools_array.is_empty() {
            body["tools"] = json!(tools_array);
        }

        // Response modalities for image generation
        if let Some(ref modalities) = options.response_modalities {
            let gc = body.as_object_mut().unwrap().entry("generationConfig").or_insert(json!({}));
            gc.as_object_mut().unwrap().remove("responseMimeType");
            gc["responseModalities"] = json!(modalities);
        }

        // Image config (aspect ratio etc.)
        if let Some(ref img_cfg) = options.image_config {
            let mut ic = json!({});
            if let Some(ref ar) = img_cfg.aspect_ratio {
                ic["aspectRatio"] = json!(ar);
            }
            let gc = body.as_object_mut().unwrap().entry("generationConfig").or_insert(json!({}));
            gc["imageConfig"] = ic;
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
                // Gemini code execution: executable code
                if let Some(ec) = part.get("executableCode") {
                    content.push(Content::ExecutableCode {
                        language: ec["language"].as_str().unwrap_or("python").to_string(),
                        code: ec["code"].as_str().unwrap_or("").to_string(),
                    });
                }
                // Gemini code execution: result
                if let Some(cer) = part.get("codeExecutionResult") {
                    content.push(Content::CodeExecutionResult {
                        outcome: cer["outcome"].as_str().unwrap_or("UNKNOWN").to_string(),
                        output: cer["output"].as_str().unwrap_or("").to_string(),
                    });
                }
                // Gemini image generation: inline image data
                if let Some(inline) = part.get("inlineData") {
                    if let (Some(mime), Some(data)) = (
                        inline["mimeType"].as_str(),
                        inline["data"].as_str(),
                    ) {
                        content.push(Content::GeneratedImage {
                            mime_type: mime.to_string(),
                            data: data.to_string(),
                        });
                    }
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

        // Parse grounding metadata
        let grounding_metadata = candidate.get("groundingMetadata").map(|gm| {
            let search_queries = gm["searchQueries"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();

            let grounding_chunks = gm["groundingChunks"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .map(|chunk| {
                            let web = chunk.get("web").unwrap_or(chunk);
                            crate::message::GroundingChunk {
                                uri: web["uri"].as_str().map(String::from),
                                title: web["title"].as_str().map(String::from),
                            }
                        })
                        .collect()
                })
                .unwrap_or_default();

            let search_entry_point = gm["searchEntryPoint"]["renderedContent"]
                .as_str()
                .map(String::from);

            crate::message::GroundingMetadata {
                search_queries,
                grounding_chunks,
                search_entry_point,
            }
        });

        Ok(GenerateResult {
            message: Message {
                role: Role::Assistant,
                content,
                name: None,
            },
            usage,
            finish_reason,
            provider_metadata: body.get("modelVersion").cloned().unwrap_or(json!(null)),
            thinking: None,
            citations: vec![],
            grounding_metadata,
        })
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl Provider for GoogleProvider {
    fn name(&self) -> &str {
        "google"
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            streaming: true,
            tool_use: true,
            vision: true,
            structured_output: true,
            grounding: true,
            code_execution: true,
            web_search: true,
            image_generation: true,
            ..Default::default()
        }
    }

    async fn generate_image(
        &self,
        prompt: &str,
        config: &crate::message::ImageGenerationConfig,
    ) -> crate::error::Result<crate::message::ImageGenerationResult> {
        let model = config.model.as_deref().unwrap_or(&self.model);
        let mut options = GenerateOptions {
            response_modalities: Some(vec!["TEXT".to_string(), "IMAGE".to_string()]),
            image_config: Some(config.clone()),
            ..Default::default()
        };
        if let Some(ref ar) = config.aspect_ratio {
            if let Some(ref mut ic) = options.image_config {
                ic.aspect_ratio = Some(ar.clone());
            }
        }

        let messages = vec![crate::message::Message::user(prompt)];
        let url = format!(
            "{}/models/{}:generateContent?key={}",
            self.base_url(),
            model,
            self.config.api_key
        );
        let body = self.build_request_body(&messages, &[], &options);

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
            .map_err(|e| GaussError::provider("google", format!("Image generation error: {e}")))?;

        let status = resp.status();
        let resp_body: serde_json::Value = resp.json().await.map_err(|e| {
            GaussError::provider("google", format!("Failed to parse image response: {e}"))
        })?;

        if !status.is_success() {
            let msg = resp_body["error"]["message"]
                .as_str()
                .unwrap_or("Unknown error");
            return Err(GaussError::provider("google", msg));
        }

        let result = self.parse_response(&resp_body)?;
        let mut images = Vec::new();

        for c in &result.message.content {
            if let Content::GeneratedImage { mime_type, data } = c {
                images.push(crate::message::GeneratedImageData {
                    url: None,
                    base64: Some(data.clone()),
                    mime_type: Some(mime_type.clone()),
                });
            }
        }

        Ok(crate::message::ImageGenerationResult {
            images,
            revised_prompt: None,
        })
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
    ) -> crate::error::Result<crate::provider::BoxStream> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn test_provider() -> GoogleProvider {
        GoogleProvider::new(
            "gemini-2.0-flash",
            ProviderConfig::new("test-key"),
        )
    }

    #[test]
    fn test_grounding_adds_google_search_tool() {
        let p = test_provider();
        let msgs = vec![Message::user("What happened today?")];
        let opts = GenerateOptions {
            grounding: true,
            ..Default::default()
        };
        let body = p.build_request_body(&msgs, &[], &opts);
        let tools = body["tools"].as_array().unwrap();
        assert!(tools.iter().any(|t| t.get("google_search").is_some()));
    }

    #[test]
    fn test_native_code_execution_adds_tool() {
        let p = test_provider();
        let msgs = vec![Message::user("Calculate 2+2")];
        let opts = GenerateOptions {
            native_code_execution: true,
            ..Default::default()
        };
        let body = p.build_request_body(&msgs, &[], &opts);
        let tools = body["tools"].as_array().unwrap();
        assert!(tools.iter().any(|t| t.get("codeExecution").is_some()));
    }

    #[test]
    fn test_grounding_and_code_execution_combined() {
        let p = test_provider();
        let msgs = vec![Message::user("Test")];
        let opts = GenerateOptions {
            grounding: true,
            native_code_execution: true,
            ..Default::default()
        };
        let body = p.build_request_body(&msgs, &[], &opts);
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 2);
    }

    #[test]
    fn test_response_modalities_for_image_gen() {
        let p = test_provider();
        let msgs = vec![Message::user("Draw a cat")];
        let opts = GenerateOptions {
            response_modalities: Some(vec!["TEXT".to_string(), "IMAGE".to_string()]),
            ..Default::default()
        };
        let body = p.build_request_body(&msgs, &[], &opts);
        let modalities = body["generationConfig"]["responseModalities"]
            .as_array()
            .unwrap();
        assert_eq!(modalities.len(), 2);
        assert_eq!(modalities[0], "TEXT");
        assert_eq!(modalities[1], "IMAGE");
    }

    #[test]
    fn test_image_config_aspect_ratio() {
        let p = test_provider();
        let msgs = vec![Message::user("Draw a landscape")];
        let opts = GenerateOptions {
            response_modalities: Some(vec!["TEXT".to_string(), "IMAGE".to_string()]),
            image_config: Some(crate::message::ImageGenerationConfig {
                aspect_ratio: Some("16:9".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let body = p.build_request_body(&msgs, &[], &opts);
        assert_eq!(body["generationConfig"]["imageConfig"]["aspectRatio"], "16:9");
    }

    #[test]
    fn test_parse_response_with_code_execution() {
        let p = test_provider();
        let resp = json!({
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "Let me calculate that."},
                        {"executableCode": {"language": "python", "code": "print(2+2)"}},
                        {"codeExecutionResult": {"outcome": "SUCCESS", "output": "4\n"}}
                    ]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20}
        });

        let result = p.parse_response(&resp).unwrap();
        assert_eq!(result.message.content.len(), 3);
        assert!(matches!(&result.message.content[1], Content::ExecutableCode { language, code } if language == "python" && code == "print(2+2)"));
        assert!(matches!(&result.message.content[2], Content::CodeExecutionResult { outcome, output } if outcome == "SUCCESS" && output == "4\n"));
    }

    #[test]
    fn test_parse_response_with_grounding_metadata() {
        let p = test_provider();
        let resp = json!({
            "candidates": [{
                "content": {
                    "parts": [{"text": "Spain won Euro 2024."}]
                },
                "finishReason": "STOP",
                "groundingMetadata": {
                    "searchQueries": ["Euro 2024 winner"],
                    "groundingChunks": [{
                        "web": {
                            "uri": "https://example.com",
                            "title": "Euro 2024 Results"
                        }
                    }],
                    "searchEntryPoint": {
                        "renderedContent": "<div>search widget</div>"
                    }
                }
            }],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 10}
        });

        let result = p.parse_response(&resp).unwrap();
        let gm = result.grounding_metadata.unwrap();
        assert_eq!(gm.search_queries, vec!["Euro 2024 winner"]);
        assert_eq!(gm.grounding_chunks.len(), 1);
        assert_eq!(gm.grounding_chunks[0].uri.as_deref(), Some("https://example.com"));
        assert_eq!(gm.grounding_chunks[0].title.as_deref(), Some("Euro 2024 Results"));
        assert!(gm.search_entry_point.is_some());
    }

    #[test]
    fn test_parse_response_with_generated_image() {
        let p = test_provider();
        let resp = json!({
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "Here is your image:"},
                        {"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgoAAAA..."}}
                    ]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 10}
        });

        let result = p.parse_response(&resp).unwrap();
        assert_eq!(result.message.content.len(), 2);
        assert!(matches!(&result.message.content[1], Content::GeneratedImage { mime_type, data } if mime_type == "image/png" && !data.is_empty()));
    }

    #[test]
    fn test_tools_with_function_declarations_and_grounding() {
        let p = test_provider();
        let msgs = vec![Message::user("Search for weather")];
        let tools = vec![
            Tool::builder("get_weather", "Get weather for a city").build(),
        ];
        let opts = GenerateOptions {
            grounding: true,
            ..Default::default()
        };
        let body = p.build_request_body(&msgs, &tools, &opts);
        let tools_arr = body["tools"].as_array().unwrap();
        // Should have function declarations tool + google_search tool
        assert_eq!(tools_arr.len(), 2);
        assert!(tools_arr[0].get("functionDeclarations").is_some());
        assert!(tools_arr[1].get("google_search").is_some());
    }

    #[test]
    fn test_capabilities_include_new_features() {
        let p = test_provider();
        let caps = p.capabilities();
        assert!(caps.grounding);
        assert!(caps.code_execution);
        assert!(caps.web_search);
        assert!(caps.image_generation);
    }
}
