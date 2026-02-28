use gauss_core::error::GaussError;
use gauss_core::message::Message;
use gauss_core::provider::openai::OpenAiProvider;
use gauss_core::provider::retry::{RetryConfig, RetryProvider};
use gauss_core::provider::{GenerateOptions, Provider, ProviderConfig};
use serde_json::json;
use std::sync::Arc;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn test_retry_on_rate_limit() {
    let mock_server = MockServer::start().await;

    // First 2 calls: rate limited, third: success
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_json(json!({
            "error": {"message": "Rate limited", "type": "rate_limit_error"}
        })))
        .up_to_n_times(2)
        .mount(&mock_server)
        .await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": {"role": "assistant", "content": "Success after retry!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3}
        })))
        .mount(&mock_server)
        .await;

    let config = ProviderConfig::new("test-key").base_url(mock_server.uri());
    let inner = Arc::new(OpenAiProvider::new("gpt-5.2", config));

    let retry_config = RetryConfig {
        max_retries: 3,
        initial_delay_ms: 10, // fast for tests
        max_delay_ms: 100,
        backoff_multiplier: 2.0,
        retry_on_rate_limit: true,
        retry_on_server_error: true,
    };

    let provider = RetryProvider::new(inner, retry_config);

    let result = provider
        .generate(&[Message::user("test")], &[], &GenerateOptions::default())
        .await
        .unwrap();

    assert_eq!(result.text().unwrap(), "Success after retry!");
}

#[tokio::test]
async fn test_retry_exhausted() {
    let mock_server = MockServer::start().await;

    // Always rate limited
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_json(json!({
            "error": {"message": "Rate limited", "type": "rate_limit_error"}
        })))
        .mount(&mock_server)
        .await;

    let config = ProviderConfig::new("test-key").base_url(mock_server.uri());
    let inner = Arc::new(OpenAiProvider::new("gpt-5.2", config));

    let retry_config = RetryConfig {
        max_retries: 2,
        initial_delay_ms: 10,
        max_delay_ms: 50,
        backoff_multiplier: 2.0,
        retry_on_rate_limit: true,
        retry_on_server_error: true,
    };

    let provider = RetryProvider::new(inner, retry_config);

    let err = provider
        .generate(&[Message::user("test")], &[], &GenerateOptions::default())
        .await
        .unwrap_err();

    assert!(matches!(err, GaussError::RateLimited { .. }));
}

#[tokio::test]
async fn test_no_retry_on_auth_error() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_json(json!({
            "error": {"message": "Invalid key", "type": "auth_error"}
        })))
        .expect(1) // should only be called once — no retry
        .mount(&mock_server)
        .await;

    let config = ProviderConfig::new("bad-key").base_url(mock_server.uri());
    let inner = Arc::new(OpenAiProvider::new("gpt-5.2", config));

    let provider = RetryProvider::wrap(inner);

    let err = provider
        .generate(&[Message::user("test")], &[], &GenerateOptions::default())
        .await
        .unwrap_err();

    assert!(matches!(err, GaussError::Authentication { .. }));
}
