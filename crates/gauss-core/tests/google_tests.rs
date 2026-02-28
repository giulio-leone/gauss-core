use gauss_core::message::Message;
use gauss_core::provider::google::GoogleProvider;
use gauss_core::provider::{GenerateOptions, Provider, ProviderConfig};
use gauss_core::tool::Tool;
use serde_json::json;
use wiremock::matchers::{method, path_regex};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn test_google_generate_text() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path_regex(r"/models/gemini-2.5-flash:generateContent.*"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello from Gemini!"}],
                    "role": "model"
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 8,
                "candidatesTokenCount": 4
            }
        })))
        .mount(&mock_server)
        .await;

    let config = ProviderConfig::new("test-key").base_url(mock_server.uri());
    let provider = GoogleProvider::new("gemini-2.5-flash", config);

    let result = provider
        .generate(
            &[Message::system("Be helpful"), Message::user("Hello!")],
            &[],
            &GenerateOptions::default(),
        )
        .await
        .unwrap();

    assert_eq!(result.text().unwrap(), "Hello from Gemini!");
    assert_eq!(result.usage.input_tokens, 8);
    assert_eq!(result.usage.output_tokens, 4);
}

#[tokio::test]
async fn test_google_generate_with_tools() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path_regex(r"/models/gemini-2.5-flash:generateContent.*"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"city": "Rome"}
                        }
                    }],
                    "role": "model"
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 15,
                "candidatesTokenCount": 10
            }
        })))
        .mount(&mock_server)
        .await;

    let config = ProviderConfig::new("test-key").base_url(mock_server.uri());
    let provider = GoogleProvider::new("gemini-2.5-flash", config);

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
async fn test_google_rate_limit() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path_regex(r"/models/.*:generateContent.*"))
        .respond_with(ResponseTemplate::new(429).set_body_json(json!({
            "error": {"code": 429, "message": "Rate limited", "status": "RESOURCE_EXHAUSTED"}
        })))
        .mount(&mock_server)
        .await;

    let config = ProviderConfig::new("test-key").base_url(mock_server.uri());
    let provider = GoogleProvider::new("gemini-2.5-flash", config);

    let err = provider
        .generate(&[Message::user("test")], &[], &GenerateOptions::default())
        .await
        .unwrap_err();

    assert!(matches!(err, gauss_core::error::GaussError::RateLimited { .. }));
}

#[tokio::test]
async fn test_google_auth_error() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path_regex(r"/models/.*:generateContent.*"))
        .respond_with(ResponseTemplate::new(403).set_body_json(json!({
            "error": {"code": 403, "message": "API key not valid", "status": "PERMISSION_DENIED"}
        })))
        .mount(&mock_server)
        .await;

    let config = ProviderConfig::new("bad-key").base_url(mock_server.uri());
    let provider = GoogleProvider::new("gemini-2.5-flash", config);

    let err = provider
        .generate(&[Message::user("test")], &[], &GenerateOptions::default())
        .await
        .unwrap_err();

    assert!(matches!(
        err,
        gauss_core::error::GaussError::Authentication { .. }
    ));
}
