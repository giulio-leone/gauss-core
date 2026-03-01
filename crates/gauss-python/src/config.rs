use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

// ============ Stream Transform ============

#[pyfunction]
pub fn agent_config_from_json(json_str: String) -> PyResult<String> {
    let config = gauss_core::config::AgentConfig::from_json(&json_str)
        .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
    config
        .to_json()
        .map_err(|e| PyRuntimeError::new_err(format!("{e}")))
}

#[pyfunction]
pub fn agent_config_resolve_env(value: String) -> String {
    gauss_core::config::resolve_env(&value)
}

// ============ AGENTS.MD & SKILL.MD Parsers ============

/// Parse an AGENTS.MD file content into an AgentSpec JSON.
#[pyfunction]
pub fn parse_agents_md(content: String) -> PyResult<String> {
    let spec = gauss_core::agents_md::parse_agents_md(&content)
        .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
    serde_json::to_string(&spec)
        .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {e}")))
}

/// Discover all AGENTS.MD files in a directory.
#[pyfunction]
pub fn discover_agents(dir: String) -> PyResult<String> {
    let specs = gauss_core::agents_md::discover_agents(&dir)
        .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
    serde_json::to_string(&specs)
        .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {e}")))
}

/// Parse a SKILL.MD file content into a SkillSpec JSON.
#[pyfunction]
pub fn parse_skill_md(content: String) -> PyResult<String> {
    let spec = gauss_core::skill_md::parse_skill_md(&content)
        .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
    serde_json::to_string(&spec)
        .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {e}")))
}
