use gauss_core::config::*;
use serde_json::json;

#[test]
fn test_config_from_json() {
    let json = r#"{
        "name": "my-agent",
        "provider": {
            "type": "openai",
            "model": "gpt-4o",
            "api_key": "sk-test"
        },
        "instructions": "You are helpful.",
        "max_steps": 5,
        "tools": [
            {
                "name": "search",
                "description": "Search the web"
            }
        ],
        "options": {
            "temperature": 0.7,
            "max_tokens": 1000
        },
        "stop_conditions": [
            {"text_generated": null}
        ]
    }"#;

    let config = AgentConfig::from_json(json).unwrap();
    assert_eq!(config.name, "my-agent");
    assert_eq!(config.provider.provider_type, "openai");
    assert_eq!(config.provider.model, "gpt-4o");
    assert_eq!(config.max_steps, 5);
    assert_eq!(config.tools.len(), 1);
    assert_eq!(config.tools[0].name, "search");
    assert_eq!(config.options.temperature, Some(0.7));
    assert_eq!(config.options.max_tokens, Some(1000));
}

#[test]
fn test_config_resolve_provider() {
    let json = r#"{
        "name": "test",
        "provider": {
            "type": "openai",
            "model": "gpt-4o",
            "api_key": "sk-plain",
            "timeout_ms": 30000
        }
    }"#;

    let config = AgentConfig::from_json(json).unwrap();
    let prov = config.resolve_provider_config();
    assert_eq!(prov.api_key, "sk-plain");
    assert_eq!(prov.timeout_ms, Some(30000));
}

#[test]
fn test_config_env_resolution() {
    // SAFETY: test-only, single-threaded context
    unsafe { std::env::set_var("GAUSS_TEST_KEY", "secret-123") };
    let resolved = resolve_env("${GAUSS_TEST_KEY}");
    assert_eq!(resolved, "secret-123");

    // Non-env values pass through
    let plain = resolve_env("plain-value");
    assert_eq!(plain, "plain-value");

    unsafe { std::env::remove_var("GAUSS_TEST_KEY") };
}

#[test]
fn test_config_build_tools() {
    let json = r#"{
        "name": "test",
        "provider": {"type": "openai", "model": "gpt-4o"},
        "tools": [
            {
                "name": "calc",
                "description": "Calculate",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expr": {"type": "string"}
                    },
                    "required": ["expr"]
                }
            }
        ]
    }"#;

    let config = AgentConfig::from_json(json).unwrap();
    let tools = config.build_tools();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "calc");
}

#[test]
fn test_config_build_options() {
    let json = r#"{
        "name": "test",
        "provider": {"type": "openai", "model": "gpt-4o"},
        "options": {
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 2048,
            "tool_choice": "required"
        }
    }"#;

    let config = AgentConfig::from_json(json).unwrap();
    let opts = config.build_options();
    assert_eq!(opts.temperature, Some(0.5));
    assert_eq!(opts.top_p, Some(0.9));
    assert_eq!(opts.max_tokens, Some(2048));
}

#[test]
fn test_config_roundtrip_json() {
    let config = AgentConfig {
        name: "roundtrip".into(),
        provider: ProviderConfigDef {
            provider_type: "anthropic".into(),
            model: "claude-3".into(),
            api_key: Some("key".into()),
            base_url: None,
            timeout_ms: None,
            max_retries: None,
        },
        instructions: Some("Be helpful".into()),
        tools: vec![],
        max_steps: 10,
        options: AgentOptionsDef::default(),
        stop_conditions: vec![],
    };

    let json_str = config.to_json().unwrap();
    let parsed = AgentConfig::from_json(&json_str).unwrap();
    assert_eq!(parsed.name, "roundtrip");
    assert_eq!(parsed.provider.provider_type, "anthropic");
}

#[test]
fn test_config_default_max_steps() {
    let json = r#"{
        "name": "test",
        "provider": {"type": "openai", "model": "gpt-4o"}
    }"#;

    let config = AgentConfig::from_json(json).unwrap();
    assert_eq!(config.max_steps, 10); // default
}

#[test]
fn test_config_invalid_json() {
    let result = AgentConfig::from_json("not valid json");
    assert!(result.is_err());
}
