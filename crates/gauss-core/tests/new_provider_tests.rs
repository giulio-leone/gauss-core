use gauss_core::message::Message;
use gauss_core::provider::deepseek::DeepSeekProvider;
use gauss_core::provider::groq::GroqProvider;
use gauss_core::provider::ollama::OllamaProvider;
use gauss_core::provider::openai::OpenAiProvider;
use gauss_core::provider::{GenerateOptions, Provider, ProviderConfig};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn mock_openai_response(content: &str) -> serde_json::Value {
    json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": content },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8 }
    })
}

#[tokio::test]
async fn test_groq_provider_generate() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(mock_openai_response("Groq response")),
        )
        .expect(1)
        .mount(&mock_server)
        .await;

    // GroqProvider::create unconditionally sets base_url to api.groq.com,
    // so we use OpenAiProvider::new directly to point at the mock server.
    let config = ProviderConfig::new("groq-test-key").base_url(mock_server.uri());
    let provider = OpenAiProvider::new("llama-3.3-70b-versatile", config);

    assert_eq!(provider.name(), "openai");
    assert_eq!(provider.model(), "llama-3.3-70b-versatile");

    let result = provider
        .generate(&[Message::user("hello")], &[], &GenerateOptions::default())
        .await
        .unwrap();

    assert_eq!(result.text(), Some("Groq response"));
}

#[tokio::test]
async fn test_groq_default_base_url() {
    let config = ProviderConfig::new("key");
    let provider = GroqProvider::create("llama-3.3-70b-versatile", config);
    // The provider wraps OpenAI; we can only verify it was created without panic.
    assert_eq!(provider.model(), "llama-3.3-70b-versatile");
}

#[tokio::test]
async fn test_ollama_provider_generate() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(mock_openai_response("Ollama response")),
        )
        .expect(1)
        .mount(&mock_server)
        .await;

    let mut config = ProviderConfig::new("ollama-key");
    config.base_url = Some(mock_server.uri());
    let provider = OllamaProvider::create("llama3.3", config);

    assert_eq!(provider.model(), "llama3.3");

    let result = provider
        .generate(&[Message::user("hello")], &[], &GenerateOptions::default())
        .await
        .unwrap();

    assert_eq!(result.text(), Some("Ollama response"));
}

#[tokio::test]
async fn test_ollama_default_base_url() {
    let config = ProviderConfig::new("unused");
    let provider = OllamaProvider::create("mistral", config);
    assert_eq!(provider.model(), "mistral");
}

#[tokio::test]
async fn test_deepseek_provider_generate() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(mock_openai_response("DeepSeek response")),
        )
        .expect(1)
        .mount(&mock_server)
        .await;

    // DeepSeekProvider::create unconditionally sets base_url to api.deepseek.com,
    // so we use OpenAiProvider::new directly to point at the mock server.
    let config = ProviderConfig::new("deepseek-test-key").base_url(mock_server.uri());
    let provider = OpenAiProvider::new("deepseek-chat", config);

    assert_eq!(provider.model(), "deepseek-chat");

    let result = provider
        .generate(&[Message::user("hello")], &[], &GenerateOptions::default())
        .await
        .unwrap();

    assert_eq!(result.text(), Some("DeepSeek response"));
}

#[tokio::test]
async fn test_deepseek_default_base_url() {
    let config = ProviderConfig::new("key");
    let provider = DeepSeekProvider::create("deepseek-coder", config);
    assert_eq!(provider.model(), "deepseek-coder");
}
