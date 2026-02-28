use gauss_core::message::Message;
use gauss_core::provider::anthropic::AnthropicProvider;
use gauss_core::provider::{GenerateOptions, Provider, ProviderConfig};
use gauss_core::tool::Tool;
use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn test_anthropic_generate_text() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header("x-api-key", "test-key"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello from Claude!"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        })))
        .mount(&mock_server)
        .await;

    let config = ProviderConfig::new("test-key").base_url(mock_server.uri());
    let provider = AnthropicProvider::new("claude-sonnet-4-20250514", config);

    let result = provider
        .generate(
            &[Message::system("Be helpful"), Message::user("Hello!")],
            &[],
            &GenerateOptions::default(),
        )
        .await
        .unwrap();

    assert_eq!(result.text().unwrap(), "Hello from Claude!");
    assert_eq!(result.usage.input_tokens, 10);
    assert_eq!(result.usage.output_tokens, 5);
}

#[tokio::test]
async fn test_anthropic_generate_with_tools() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_456",
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": "toolu_123",
                "name": "get_weather",
                "input": {"city": "Rome"}
            }],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 15}
        })))
        .mount(&mock_server)
        .await;

    let config = ProviderConfig::new("test-key").base_url(mock_server.uri());
    let provider = AnthropicProvider::new("claude-sonnet-4-20250514", config);

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

    let tc = result.tool_calls();
    assert_eq!(tc.len(), 1);
    assert_eq!(tc[0].1, "get_weather");
    assert_eq!(tc[0].2["city"], "Rome");
}

#[tokio::test]
async fn test_anthropic_rate_limit() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(429).set_body_json(json!({
            "type": "error",
            "error": {"type": "rate_limit_error", "message": "Rate limited"}
        })))
        .mount(&mock_server)
        .await;

    let config = ProviderConfig::new("test-key").base_url(mock_server.uri());
    let provider = AnthropicProvider::new("claude-sonnet-4-20250514", config);

    let err = provider
        .generate(&[Message::user("test")], &[], &GenerateOptions::default())
        .await
        .unwrap_err();

    assert!(matches!(err, gauss_core::error::GaussError::RateLimited { .. }));
}

#[tokio::test]
async fn test_anthropic_auth_error() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(401).set_body_json(json!({
            "type": "error",
            "error": {"type": "authentication_error", "message": "Invalid API key"}
        })))
        .mount(&mock_server)
        .await;

    let config = ProviderConfig::new("bad-key").base_url(mock_server.uri());
    let provider = AnthropicProvider::new("claude-sonnet-4-20250514", config);

    let err = provider
        .generate(&[Message::user("test")], &[], &GenerateOptions::default())
        .await
        .unwrap_err();

    assert!(matches!(
        err,
        gauss_core::error::GaussError::Authentication { .. }
    ));
}
