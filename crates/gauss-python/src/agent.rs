use crate::provider::get_provider;
use crate::registry::py_err;
use crate::types::{parse_messages, parse_reasoning_effort};
use gauss_core::agent::{Agent as RustAgent, StopCondition};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde_json::json;

/// Run an agent. Returns JSON string.
#[pyfunction]
#[pyo3(signature = (name, provider_handle, messages_json, options_json=None))]
pub fn agent_run(
    py: Python<'_>,
    name: String,
    provider_handle: u32,
    messages_json: String,
    options_json: Option<String>,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let provider = get_provider(provider_handle)?;
        let rust_msgs = parse_messages(&messages_json)?;
        let mut builder = RustAgent::builder(name, provider);

        if let Some(opts_str) = options_json {
            let opts: serde_json::Value = serde_json::from_str(&opts_str)
                .map_err(|e| PyRuntimeError::new_err(format!("Invalid options JSON: {e}")))?;

            if let Some(instructions) = opts["instructions"].as_str() {
                builder = builder.instructions(instructions);
            }
            if let Some(max_steps) = opts["max_steps"].as_u64() {
                builder = builder.max_steps(max_steps as usize);
            }
            if let Some(temp) = opts["temperature"].as_f64() {
                builder = builder.temperature(temp);
            }
            if let Some(tp) = opts["top_p"].as_f64() {
                builder = builder.top_p(tp);
            }
            if let Some(mt) = opts["max_tokens"].as_u64() {
                builder = builder.max_tokens(mt as u32);
            }
            if let Some(stop_tool) = opts["stop_on_tool"].as_str() {
                builder = builder.stop_when(StopCondition::HasToolCall(stop_tool.to_string()));
            }
            if let Some(schema) = opts.get("output_schema").filter(|s| !s.is_null()) {
                builder = builder.output_schema(schema.clone());
            }
            if let Some(budget) = opts["thinking_budget"].as_u64() {
                builder = builder.thinking_budget(budget as u32);
            }
            if let Some(effort_str) = opts["reasoning_effort"].as_str() {
                if let Some(effort) = parse_reasoning_effort(effort_str) {
                    builder = builder.reasoning_effort(effort);
                }
            }
            if opts["cache_control"].as_bool().unwrap_or(false) {
                builder = builder.cache_control(true);
            }

            // Code execution
            if let Some(ce) = opts.get("code_execution").filter(|v| !v.is_null()) {
                let ce_config = if ce.is_boolean() && ce.as_bool() == Some(true) {
                    gauss_core::code_execution::CodeExecutionConfig::all()
                } else if ce.is_object() {
                    let sandbox = match ce["sandbox"].as_str() {
                        Some("strict") => gauss_core::code_execution::SandboxConfig::strict(),
                        Some("permissive") => {
                            gauss_core::code_execution::SandboxConfig::permissive()
                        }
                        _ => gauss_core::code_execution::SandboxConfig::default(),
                    };
                    gauss_core::code_execution::CodeExecutionConfig {
                        python: ce["python"].as_bool().unwrap_or(true),
                        javascript: ce["javascript"].as_bool().unwrap_or(true),
                        bash: ce["bash"].as_bool().unwrap_or(true),
                        timeout: std::time::Duration::from_secs(
                            ce["timeout_secs"].as_u64().unwrap_or(30),
                        ),
                        working_dir: ce["working_dir"].as_str().map(|s| s.to_string()),
                        env: Vec::new(),
                        sandbox,
                        interpreters: std::collections::HashMap::new(),
                    }
                } else {
                    gauss_core::code_execution::CodeExecutionConfig::all()
                };

                let unified = ce.get("unified").and_then(|v| v.as_bool()).unwrap_or(false);
                if unified {
                    builder = builder.code_execution_unified(ce_config);
                } else {
                    builder = builder.code_execution(ce_config);
                }
            }

            // Grounding (Gemini)
            if opts
                .get("grounding")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                builder = builder.grounding(true);
            }
            // Native code execution (Gemini code interpreter)
            if opts
                .get("native_code_execution")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                builder = builder.native_code_execution(true);
            }
            // Response modalities
            if let Some(modalities) = opts.get("response_modalities").and_then(|v| v.as_array()) {
                let mods: Vec<String> = modalities
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
                builder = builder.response_modalities(mods);
            }
        }

        let agent = builder.build();
        let output = agent
            .run(rust_msgs)
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("Agent error: {e}")))?;

        let agent_citations: Vec<serde_json::Value> = output
            .citations
            .iter()
            .map(|c| {
                json!({
                    "type": c.citation_type,
                    "cited_text": c.cited_text,
                    "document_title": c.document_title,
                    "start": c.start,
                    "end": c.end,
                })
            })
            .collect();

        let result = json!({
            "text": output.text,
            "thinking": output.thinking,
            "citations": agent_citations,
            "grounding_metadata": output.grounding_metadata,
            "steps": output.steps,
            "usage": {
                "input_tokens": output.usage.input_tokens,
                "output_tokens": output.usage.output_tokens,
            },
            "structured_output": output.structured_output,
        });

        serde_json::to_string(&result).map_err(|e| py_err(format!("Serialize error: {e}")))
    })
}
