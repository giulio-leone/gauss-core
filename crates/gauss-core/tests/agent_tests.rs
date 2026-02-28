use gauss_core::agent::Agent;
use gauss_core::message::Message;
use gauss_core::provider::openai::OpenAiProvider;
use gauss_core::provider::{ProviderConfig, ReasoningEffort};
use gauss_core::tool::Tool;
use serde_json::json;
use std::sync::Arc;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn test_agent_simple_text() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": {"role": "assistant", "content": "Hello from the agent!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 15, "completion_tokens": 6}
        })))
        .mount(&mock_server)
        .await;

    let config = ProviderConfig::new("test-key").base_url(mock_server.uri());
    let provider = Arc::new(OpenAiProvider::new("gpt-5.2", config));

    let agent = Agent::builder("test-agent", provider)
        .instructions("You are a helpful assistant.")
        .temperature(0.7)
        .build();

    let output = agent.run(vec![Message::user("Hi!")]).await.unwrap();

    assert_eq!(output.text, "Hello from the agent!");
    assert_eq!(output.steps, 1);
    assert_eq!(output.usage.input_tokens, 15);
    assert_eq!(output.usage.output_tokens, 6);
}

#[tokio::test]
async fn test_agent_with_tool_use() {
    let mock_server = MockServer::start().await;

    // First call: model requests tool use
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": "{\"expression\":\"2+2\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10}
        })))
        .up_to_n_times(1)
        .mount(&mock_server)
        .await;

    // Second call: model responds with final text
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": {"role": "assistant", "content": "The result of 2+2 is 4."},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 30, "completion_tokens": 8}
        })))
        .mount(&mock_server)
        .await;

    let config = ProviderConfig::new("test-key").base_url(mock_server.uri());
    let provider = Arc::new(OpenAiProvider::new("gpt-5.2", config));

    let tool = Tool::builder("calculator", "Evaluates math expressions")
        .parameters_json(json!({
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]
        }))
        .execute(|args| async move {
            let expr = args["expression"].as_str().unwrap_or("0");
            // Simple mock calculation
            let result = if expr == "2+2" { 4 } else { 0 };
            Ok(json!({"result": result}))
        })
        .build();

    let agent = Agent::builder("calc-agent", provider)
        .instructions("You can use the calculator tool.")
        .tool(tool)
        .max_steps(5)
        .build();

    let output = agent
        .run(vec![Message::user("What is 2+2?")])
        .await
        .unwrap();

    assert_eq!(output.text, "The result of 2+2 is 4.");
    assert_eq!(output.steps, 2);
    assert!(output.usage.input_tokens > 0);

    // Verify tool was called in step 0
    let step0 = &output.step_results[0];
    assert_eq!(step0.tool_calls.len(), 1);
    assert_eq!(step0.tool_calls[0].name, "calculator");
    assert_eq!(step0.tool_results.len(), 1);
    assert!(!step0.tool_results[0].is_error);
}

#[tokio::test]
async fn test_agent_builder_fluent() {
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
        .mount(&mock_server)
        .await;

    let config = ProviderConfig::new("test-key").base_url(mock_server.uri());
    let provider = Arc::new(OpenAiProvider::new("gpt-5.2", config));

    let agent = Agent::builder("fluent-agent", provider)
        .instructions("Be helpful")
        .temperature(0.5)
        .top_p(0.9)
        .max_tokens(1000)
        .seed(42)
        .reasoning_effort(ReasoningEffort::High)
        .frequency_penalty(0.1)
        .presence_penalty(0.2)
        .max_steps(3)
        .build();

    let output = agent.run(vec![Message::user("test")]).await.unwrap();
    assert_eq!(output.text, "test");
}

#[tokio::test]
async fn test_agent_max_steps_limit() {
    let mock_server = MockServer::start().await;

    // Always return tool calls to force max steps
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_loop",
                        "type": "function",
                        "function": {
                            "name": "infinite",
                            "arguments": "{}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        })))
        .mount(&mock_server)
        .await;

    let config = ProviderConfig::new("test-key").base_url(mock_server.uri());
    let provider = Arc::new(OpenAiProvider::new("gpt-5.2", config));

    let tool = Tool::builder("infinite", "Never-ending tool")
        .execute(|_| async move { Ok(json!({"status": "again"})) })
        .build();

    let agent = Agent::builder("loop-agent", provider)
        .tool(tool)
        .max_steps(3)
        .build();

    let output = agent.run(vec![Message::user("loop")]).await.unwrap();
    assert_eq!(output.steps, 3);
}
