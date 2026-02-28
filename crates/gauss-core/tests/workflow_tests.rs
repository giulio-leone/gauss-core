use gauss_core::agent::Agent;
use gauss_core::message::Message;
use gauss_core::provider::openai::OpenAiProvider;
use gauss_core::provider::{GenerateOptions, Provider, ProviderConfig};
use gauss_core::workflow::{StepOutput, Workflow};
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
async fn test_workflow_sequential_two_steps() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(mock_openai_response("Step A done")),
        )
        .up_to_n_times(1)
        .mount(&mock_server)
        .await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(mock_openai_response("Step B done")),
        )
        .mount(&mock_server)
        .await;

    let provider = make_provider(&mock_server.uri());

    let agent_a = Agent::builder("agent-a", provider.clone() as Arc<dyn Provider>)
        .instructions("You are step A")
        .build();
    let agent_b = Agent::builder("agent-b", provider as Arc<dyn Provider>)
        .instructions("You are step B")
        .build();

    let workflow = Workflow::builder()
        .agent_step("step_a", agent_a, |_outputs| {
            vec![Message::user("start")]
        })
        .agent_step("step_b", agent_b, |outputs| {
            let prev = outputs
                .get("step_a")
                .map(|o| o.text.clone())
                .unwrap_or_default();
            vec![Message::user(format!("continue from: {prev}"))]
        })
        .dependency("step_b", "step_a")
        .build();

    let results = workflow.run(vec![Message::user("begin")]).await.unwrap();

    assert!(results.contains_key("step_a"));
    assert!(results.contains_key("step_b"));
    assert_eq!(results.len(), 2);
}

#[tokio::test]
async fn test_workflow_dag_three_steps() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(mock_openai_response("Test response")),
        )
        .mount(&mock_server)
        .await;

    let provider = make_provider(&mock_server.uri());

    let agent_a = Agent::builder("agent-a", provider.clone() as Arc<dyn Provider>)
        .instructions("Step A")
        .build();
    let agent_b = Agent::builder("agent-b", provider.clone() as Arc<dyn Provider>)
        .instructions("Step B")
        .build();
    let agent_c = Agent::builder("agent-c", provider as Arc<dyn Provider>)
        .instructions("Step C")
        .build();

    let workflow = Workflow::builder()
        .agent_step("step_a", agent_a, |_| vec![Message::user("a")])
        .agent_step("step_b", agent_b, |_| vec![Message::user("b")])
        .agent_step("step_c", agent_c, |outputs| {
            let a_text = outputs
                .get("step_a")
                .map(|o| o.text.as_str())
                .unwrap_or("");
            let b_text = outputs
                .get("step_b")
                .map(|o| o.text.as_str())
                .unwrap_or("");
            vec![Message::user(format!("merge: {a_text} + {b_text}"))]
        })
        .dependency("step_c", "step_a")
        .dependency("step_c", "step_b")
        .build();

    let results = workflow.run(vec![Message::user("go")]).await.unwrap();

    assert_eq!(results.len(), 3);
    assert!(results.contains_key("step_a"));
    assert!(results.contains_key("step_b"));
    assert!(results.contains_key("step_c"));
}

#[tokio::test]
async fn test_workflow_deadlock_detection() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(mock_openai_response("unreachable")),
        )
        .mount(&mock_server)
        .await;

    let provider = make_provider(&mock_server.uri());

    let agent_a = Agent::builder("agent-a", provider.clone() as Arc<dyn Provider>)
        .instructions("A")
        .build();
    let agent_b = Agent::builder("agent-b", provider as Arc<dyn Provider>)
        .instructions("B")
        .build();

    let workflow = Workflow::builder()
        .agent_step("step_a", agent_a, |_| vec![Message::user("a")])
        .agent_step("step_b", agent_b, |_| vec![Message::user("b")])
        .dependency("step_a", "step_b")
        .dependency("step_b", "step_a")
        .build();

    let result = workflow.run(vec![Message::user("deadlock")]).await;

    // Circular deps mean neither step is an entry point, so neither executes.
    // The engine either returns an error or an empty result set.
    match result {
        Err(e) => {
            let err = e.to_string();
            assert!(
                err.contains("deadlock") || err.contains("circular"),
                "Expected deadlock error, got: {err}"
            );
        }
        Ok(outputs) => {
            assert!(
                outputs.is_empty(),
                "Expected no steps to execute due to circular deps, got {} outputs",
                outputs.len()
            );
        }
    }
}
