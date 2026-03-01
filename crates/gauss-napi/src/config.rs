use gauss_core::stream_transform;
use napi::bindgen_prelude::*;

// ============ Stream Transform ============

#[napi]
pub fn parse_partial_json(text: String) -> Option<String> {
    stream_transform::parse_partial_json(&text).map(|v| v.to_string())
}

// ============ Agent Config ============

#[napi]
pub fn agent_config_from_json(json_str: String) -> Result<String> {
    let config = gauss_core::config::AgentConfig::from_json(&json_str)
        .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
    config
        .to_json()
        .map_err(|e| napi::Error::from_reason(format!("{e}")))
}

#[napi]
pub fn agent_config_resolve_env(value: String) -> String {
    gauss_core::config::resolve_env(&value)
}

// ============ AGENTS.MD & SKILL.MD Parsers ============

#[napi]
pub fn parse_agents_md(content: String) -> Result<serde_json::Value> {
    let spec = gauss_core::agents_md::parse_agents_md(&content)
        .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
    serde_json::to_value(&spec).map_err(|e| napi::Error::from_reason(format!("{e}")))
}

#[napi]
pub fn discover_agents(dir: String) -> Result<serde_json::Value> {
    let specs = gauss_core::agents_md::discover_agents(&dir)
        .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
    serde_json::to_value(&specs).map_err(|e| napi::Error::from_reason(format!("{e}")))
}

#[napi]
pub fn parse_skill_md(content: String) -> Result<serde_json::Value> {
    let spec = gauss_core::skill_md::parse_skill_md(&content)
        .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
    serde_json::to_value(&spec).map_err(|e| napi::Error::from_reason(format!("{e}")))
}
