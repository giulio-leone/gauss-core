use gauss_core::message::Message;
use gauss_core::provider::openai::OpenAiProvider;
use gauss_core::provider::{GenerateOptions, Provider, ProviderConfig, ReasoningEffort};
use gauss_core::tool::Tool;
use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn test_openai_generate_text() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("Authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! I'm an AI assistant."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18
            }
        })))
        .mount(&mock_server)
        .await;

    let config = ProviderConfig::new("test-key").base_url(mock_server.uri());
    let provider = OpenAiProvider::new("gpt-5.2", config);

    assert_eq!(provider.name(), "openai");
    assert_eq!(provider.model(), "gpt-5.2");

    let messages = vec![Message::user("Hello!")];
    let result = provider
        .generate(&messages, &[], &GenerateOptions::default())
        .await
        .unwrap();

    assert_eq!(result.text(), Some("Hello! I'm an AI assistant."));
    assert_eq!(result.usage.input_tokens, 10);
    assert_eq!(result.usage.output_tokens, 8);
}

#[tokio::test]
async fn test_openai_generate_with_tools() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-456",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"Rome\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 15}
        })))
        .mount(&mock_server)
        .await;

    let config = ProviderConfig::new("test-key").base_url(mock_server.uri());
    let provider = OpenAiProvider::new("gpt-5.2", config);

    let tool = Tool::builder("get_weather", "Get weather for a city")
        .parameters_json(json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }))
        .build();

    let result = provider
        .generate(
            &[Message::user("Weather in Rome?")],
            &[tool],
            &GenerateOptions::default(),
        )
        .await
        .unwrap();

    let calls = result.tool_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].1, "get_weather");
    assert_eq!(calls[0].2["city"], "Rome");
}

#[tokio::test]
async fn test_openai_generate_options() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": {"role": "assistant", "content": "test"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1}
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let config = ProviderConfig::new("test-key").base_url(mock_server.uri());
    let provider = OpenAiProvider::new("gpt-5.2", config);

    let options = GenerateOptions {
        temperature: Some(0.7),
        top_p: Some(0.9),
        max_tokens: Some(100),
        seed: Some(42),
        reasoning_effort: Some(ReasoningEffort::High),
        frequency_penalty: Some(0.5),
        presence_penalty: Some(0.3),
        ..Default::default()
    };

    let result = provider
        .generate(&[Message::user("test")], &[], &options)
        .await
        .unwrap();

    assert_eq!(result.text(), Some("test"));
}

#[tokio::test]
async fn test_openai_rate_limit_error() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_json(json!({
            "error": {
                "message": "Rate limit exceeded",
                "type": "rate_limit_error"
            }
        })))
        .mount(&mock_server)
        .await;

    let config = ProviderConfig::new("test-key").base_url(mock_server.uri());
    let provider = OpenAiProvider::new("gpt-5.2", config);

    let result = provider
        .generate(&[Message::user("test")], &[], &GenerateOptions::default())
        .await;

    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        gauss_core::error::GaussError::RateLimited { .. }
    ));
}

#[tokio::test]
async fn test_openai_auth_error() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_json(json!({
            "error": {"message": "Invalid API key"}
        })))
        .mount(&mock_server)
        .await;

    let config = ProviderConfig::new("bad-key").base_url(mock_server.uri());
    let provider = OpenAiProvider::new("gpt-5.2", config);

    let result = provider
        .generate(&[Message::user("test")], &[], &GenerateOptions::default())
        .await;

    assert!(matches!(
        result.unwrap_err(),
        gauss_core::error::GaussError::Authentication { .. }
    ));
}

#[tokio::test]
async fn test_openai_structured_output() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "{\"name\":\"Rome\",\"population\":2873000}"
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 15}
        })))
        .mount(&mock_server)
        .await;

    let config = ProviderConfig::new("test-key").base_url(mock_server.uri());
    let provider = OpenAiProvider::new("gpt-5.2", config);

    let options = GenerateOptions {
        output_schema: Some(json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "population": {"type": "integer"}
            },
            "required": ["name", "population"]
        })),
        ..Default::default()
    };

    let result = provider
        .generate(&[Message::user("Tell me about Rome")], &[], &options)
        .await
        .unwrap();

    let text = result.text().unwrap();
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert_eq!(parsed["name"], "Rome");
    assert_eq!(parsed["population"], 2873000);
}
