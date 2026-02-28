use gauss_core::agent::Agent;
use gauss_core::network::*;
use gauss_core::provider::ProviderConfig;
use gauss_core::provider::openai::OpenAiProvider;
use std::sync::Arc;

fn make_agent(name: &str) -> Agent {
    let config = ProviderConfig::new("test-key");
    let provider = Arc::new(OpenAiProvider::new("gpt-4", config));
    Agent::builder(name, provider).build()
}

#[test]
fn test_agent_card_creation() {
    let card = AgentCard {
        name: "researcher".into(),
        description: "Searches and analyzes information".into(),
        capabilities: vec!["search".into(), "summarize".into()],
        url: None,
        input_modes: vec![],
        output_modes: vec![],
    };

    assert_eq!(card.name, "researcher");
    assert_eq!(card.capabilities.len(), 2);
}

#[test]
fn test_agent_network_add_and_route() {
    let mut network = AgentNetwork::new();

    let card = AgentCard {
        name: "coder".into(),
        description: "Writes code".into(),
        capabilities: vec!["code".into(), "debug".into()],
        url: None,
        input_modes: vec![],
        output_modes: vec![],
    };

    let node = AgentNode {
        agent: make_agent("coder"),
        card,
        connections: vec![],
    };

    network.add_agent(node);

    let result = network.route(&["code".into()]);
    assert_eq!(result, Some("coder"));

    let result = network.route(&["dancing".into()]);
    assert!(result.is_none());
}

#[test]
fn test_network_multi_agent_routing() {
    let mut network = AgentNetwork::new();

    network.add_agent(AgentNode {
        agent: make_agent("researcher"),
        card: AgentCard {
            name: "researcher".into(),
            description: "Research".into(),
            capabilities: vec!["search".into(), "analyze".into()],
            url: None,
            input_modes: vec![],
            output_modes: vec![],
        },
        connections: vec![],
    });

    network.add_agent(AgentNode {
        agent: make_agent("coder"),
        card: AgentCard {
            name: "coder".into(),
            description: "Code".into(),
            capabilities: vec!["code".into(), "test".into(), "analyze".into()],
            url: None,
            input_modes: vec![],
            output_modes: vec![],
        },
        connections: vec![],
    });

    // "analyze" + "code" → coder has 2 matches
    let result = network.route(&["analyze".into(), "code".into()]);
    assert_eq!(result, Some("coder"));

    // "search" → only researcher
    let result = network.route(&["search".into()]);
    assert_eq!(result, Some("researcher"));
}

#[test]
fn test_network_message_types() {
    let msg = NetworkMessage {
        from: "agent-a".into(),
        to: "agent-b".into(),
        content: "summarize this".into(),
        metadata: Default::default(),
    };

    assert_eq!(msg.from, "agent-a");
    assert_eq!(msg.to, "agent-b");
    assert_eq!(msg.content, "summarize this");
}

#[test]
fn test_delegation_result() {
    let result = DelegationResult {
        agent_name: "worker-1".into(),
        result_text: "Task completed successfully".into(),
        success: true,
        error: None,
    };

    assert!(result.success);
    assert_eq!(result.result_text, "Task completed successfully");
}
