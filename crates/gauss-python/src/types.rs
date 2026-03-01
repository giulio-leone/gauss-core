use gauss_core::message::Message as RustMessage;
use gauss_core::provider::ReasoningEffort;
use pyo3::exceptions::PyRuntimeError;
use pyo3::PyResult;

pub fn parse_messages(messages_json: &str) -> PyResult<Vec<RustMessage>> {
    let js_msgs: Vec<serde_json::Value> = serde_json::from_str(messages_json)
        .map_err(|e| PyRuntimeError::new_err(format!("Invalid messages JSON: {e}")))?;
    Ok(js_msgs
        .iter()
        .map(|m| {
            let role = m["role"].as_str().unwrap_or("user");
            let content = m["content"].as_str().unwrap_or("");
            match role {
                "system" => RustMessage::system(content),
                "assistant" => RustMessage::assistant(content),
                _ => RustMessage::user(content),
            }
        })
        .collect())
}

pub fn parse_reasoning_effort(s: &str) -> Option<ReasoningEffort> {
    match s.to_lowercase().as_str() {
        "low" => Some(ReasoningEffort::Low),
        "medium" => Some(ReasoningEffort::Medium),
        "high" => Some(ReasoningEffort::High),
        _ => None,
    }
}
