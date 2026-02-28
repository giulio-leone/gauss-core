use gauss_core::agent::Agent;
use gauss_core::message::Message;
use gauss_core::provider::openai::OpenAiProvider;
use gauss_core::provider::{Provider, ProviderConfig};
use gauss_core::team::{Strategy, Team};
use serde_json::json;
use std::sync::Arc;
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

fn make_provider(uri: &str) -> Arc<OpenAiProvider> {
    let config = ProviderConfig::new("test-key").base_url(uri);
    Arc::new(OpenAiProvider::new("gpt-test", config))
}

#[tokio::test]
async fn test_team_sequential_strategy() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(mock_openai_response("Agent 1 output")),
        )
        .up_to_n_times(1)
        .mount(&mock_server)
        .await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(mock_openai_response("Agent 2 output")),
        )
        .mount(&mock_server)
        .await;

    let provider = make_provider(&mock_server.uri());

    let agent1 = Agent::builder("agent-1", provider.clone() as Arc<dyn Provider>)
        .instructions("First agent")
        .build();
    let agent2 = Agent::builder("agent-2", provider as Arc<dyn Provider>)
        .instructions("Second agent")
        .build();

    let team = Team::builder("test-team")
        .strategy(Strategy::Sequential)
        .agent(agent1)
        .agent(agent2)
        .build();

    let output = team.run(vec![Message::user("hello")]).await.unwrap();

    assert_eq!(output.results.len(), 2);
    assert_eq!(output.final_text, "Agent 2 output");
}

#[tokio::test]
async fn test_team_parallel_strategy() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(mock_openai_response("Parallel response")),
        )
        .mount(&mock_server)
        .await;

    let provider = make_provider(&mock_server.uri());

    let agent1 = Agent::builder("par-1", provider.clone() as Arc<dyn Provider>)
        .instructions("Parallel agent 1")
        .build();
    let agent2 = Agent::builder("par-2", provider as Arc<dyn Provider>)
        .instructions("Parallel agent 2")
        .build();

    let team = Team::builder("parallel-team")
        .strategy(Strategy::Parallel)
        .agent(agent1)
        .agent(agent2)
        .build();

    let output = team.run(vec![Message::user("go")]).await.unwrap();

    assert_eq!(output.results.len(), 2);
    assert!(output.final_text.contains("Parallel response"));
}

#[tokio::test]
async fn test_team_empty_returns_error() {
    let team = Team::builder("empty-team")
        .strategy(Strategy::Sequential)
        .build();

    let result = team.run(vec![Message::user("hello")]).await;

    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("no agents"),
        "Expected 'no agents' error, got: {err}"
    );
}
