use crate::provider::get_provider;
use crate::registry::py_err;
use crate::types::{parse_messages, parse_reasoning_effort};
use gauss_core::agent::{Agent as RustAgent, AgentBuilder, AgentStreamEvent, StopCondition};
use gauss_core::tool::Tool as RustTool;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde_json::json;
use std::sync::Arc;

/// Apply agent options from a JSON string onto the builder.
fn apply_options(
    mut builder: AgentBuilder,
    options_json: &Option<String>,
) -> PyResult<AgentBuilder> {
    let opts_str = match options_json {
        Some(s) => s,
        None => return Ok(builder),
    };

    let opts: serde_json::Value = serde_json::from_str(opts_str)
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
                Some("permissive") => gauss_core::code_execution::SandboxConfig::permissive(),
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

    Ok(builder)
}

/// Serialize agent output to JSON string.
fn serialize_agent_output(output: &gauss_core::agent::AgentOutput) -> PyResult<String> {
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
}

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
        let builder = RustAgent::builder(name, provider);
        let builder = apply_options(builder, &options_json)?;

        let agent = builder.build();
        let output = agent
            .run(rust_msgs)
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("Agent error: {e}")))?;

        serialize_agent_output(&output)
    })
}

/// Run an agent with tool execution delegated to Python. Returns JSON string.
#[pyfunction]
#[pyo3(signature = (name, provider_handle, messages_json, options_json=None, tool_executor=None))]
pub fn agent_run_with_tool_executor(
    py: Python<'_>,
    name: String,
    provider_handle: u32,
    messages_json: String,
    options_json: Option<String>,
    tool_executor: Option<PyObject>,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let provider = get_provider(provider_handle)?;
        let rust_msgs = parse_messages(&messages_json)?;
        let mut builder = RustAgent::builder(name, provider);
        builder = apply_options(builder, &options_json)?;

        // Parse tools from options and attach Python executor
        let exec_arc: Option<Arc<PyObject>> = tool_executor.map(Arc::new);
        if let Some(exec) = &exec_arc {
            let tools_value = options_json
                .as_ref()
                .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
                .and_then(|v| v.get("tools").cloned());

            if let Some(serde_json::Value::Array(tool_defs)) = tools_value {
                for td in &tool_defs {
                    let td_name = td["name"].as_str().unwrap_or("unknown").to_string();
                    let td_desc = td["description"].as_str().unwrap_or("").to_string();

                    let mut tool_builder = RustTool::builder(&td_name, &td_desc);
                    if let Some(params) = td.get("parameters").filter(|p| !p.is_null()) {
                        tool_builder = tool_builder.parameters_json(params.clone());
                    }

                    let tool_exec = exec.clone();
                    let tool_name_owned = td_name.clone();

                    tool_builder =
                        tool_builder.execute(move |args: serde_json::Value| {
                            let exec = tool_exec.clone();
                            let tn = tool_name_owned.clone();
                            async move {
                                let call_json =
                                    serde_json::to_string(&json!({"tool": tn, "args": args}))
                                        .map_err(|e| {
                                            gauss_core::error::GaussError::tool(
                                                &tn,
                                                format!("Serialize: {e}"),
                                            )
                                        })?;

                                let (sync_result, async_future) = Python::with_gil(
                                    |py| -> PyResult<(Option<String>, Option<_>)> {
                                        let result = exec.call1(py, (call_json,))?;
                                        let inspect = py.import("inspect")?;
                                        let is_coro = inspect
                                            .call_method1(
                                                "isawaitable",
                                                (result.bind(py),),
                                            )?
                                            .is_truthy()?;
                                        if is_coro {
                                            let fut =
                                                pyo3_async_runtimes::tokio::into_future(
                                                    result.into_bound(py),
                                                )?;
                                            Ok((None, Some(fut)))
                                        } else {
                                            let s = result.extract::<String>(py)?;
                                            Ok((Some(s), None))
                                        }
                                    },
                                )
                                .map_err(|e| {
                                    gauss_core::error::GaussError::tool(
                                        &tn,
                                        format!("Python call error: {e}"),
                                    )
                                })?;

                                let result_str = if let Some(s) = sync_result {
                                    s
                                } else if let Some(fut) = async_future {
                                    let py_obj = fut.await.map_err(|e| {
                                        gauss_core::error::GaussError::tool(
                                            &tn,
                                            format!("Python async error: {e}"),
                                        )
                                    })?;
                                    Python::with_gil(|py| py_obj.extract::<String>(py)).map_err(
                                        |e| {
                                            gauss_core::error::GaussError::tool(
                                                &tn,
                                                format!("Python extract error: {e}"),
                                            )
                                        },
                                    )?
                                } else {
                                    unreachable!()
                                };

                                serde_json::from_str(&result_str).map_err(|e| {
                                    gauss_core::error::GaussError::tool(
                                        &tn,
                                        format!("Deserialize: {e}"),
                                    )
                                })
                            }
                        });

                    builder = builder.tool(tool_builder.build());
                }
            }
        }

        let agent = builder.build();
        let output = agent
            .run(rust_msgs)
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("Agent error: {e}")))?;

        serialize_agent_output(&output)
    })
}

/// Stream an agent execution, pushing events to Python via callback. Returns JSON string.
#[pyfunction]
#[pyo3(signature = (name, provider_handle, messages_json, stream_callback, options_json=None))]
pub fn agent_stream(
    py: Python<'_>,
    name: String,
    provider_handle: u32,
    messages_json: String,
    stream_callback: PyObject,
    options_json: Option<String>,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        use futures::StreamExt;

        let provider = get_provider(provider_handle)?;
        let rust_msgs = parse_messages(&messages_json)?;
        let builder = RustAgent::builder(name, provider);
        let builder = apply_options(builder, &options_json)?;

        let agent = builder.build();
        let mut stream = agent
            .run_stream(rust_msgs)
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("Stream init error: {e}")))?;

        let mut final_text = String::new();
        let mut final_steps = 0u32;
        let mut final_input_tokens = 0u32;
        let mut final_output_tokens = 0u32;

        let cb = Arc::new(stream_callback);

        while let Some(event) = stream.next().await {
            let event_json = match event {
                Ok(AgentStreamEvent::TextDelta { step, delta }) => {
                    serde_json::to_string(&json!({
                        "type": "text_delta",
                        "step": step,
                        "delta": delta,
                    }))
                }
                Ok(AgentStreamEvent::StepStart { step }) => {
                    serde_json::to_string(&json!({
                        "type": "step_start",
                        "step": step,
                    }))
                }
                Ok(AgentStreamEvent::ToolCallDelta { step, index }) => {
                    serde_json::to_string(&json!({
                        "type": "tool_call_delta",
                        "step": step,
                        "index": index,
                    }))
                }
                Ok(AgentStreamEvent::ToolResult {
                    step,
                    tool_name,
                    result,
                    is_error,
                }) => serde_json::to_string(&json!({
                    "type": "tool_result",
                    "step": step,
                    "tool_name": tool_name,
                    "result": result,
                    "is_error": is_error,
                })),
                Ok(AgentStreamEvent::StepFinish {
                    step,
                    finish_reason,
                    has_tool_calls,
                }) => serde_json::to_string(&json!({
                    "type": "step_finish",
                    "step": step,
                    "finish_reason": format!("{:?}", finish_reason),
                    "has_tool_calls": has_tool_calls,
                })),
                Ok(AgentStreamEvent::Done { text, steps, usage }) => {
                    final_text = text.clone();
                    final_steps = steps as u32;
                    final_input_tokens = usage.input_tokens as u32;
                    final_output_tokens = usage.output_tokens as u32;

                    serde_json::to_string(&json!({
                        "type": "done",
                        "text": text,
                        "steps": final_steps,
                        "input_tokens": final_input_tokens,
                        "output_tokens": final_output_tokens,
                    }))
                }
                Ok(AgentStreamEvent::RawEvent { step, event }) => {
                    serde_json::to_string(&json!({
                        "type": "raw_event",
                        "step": step,
                        "event": format!("{:?}", event),
                    }))
                }
                Err(e) => {
                    let err_json = serde_json::to_string(&json!({
                        "type": "error",
                        "error": format!("{e}"),
                    }))
                    .unwrap_or_default();

                    let _ = Python::with_gil(|py| cb.call1(py, (err_json,)));
                    return Err(PyRuntimeError::new_err(format!("Stream error: {e}")));
                }
            };

            if let Ok(json_str) = event_json {
                let _ = Python::with_gil(|py| cb.call1(py, (json_str,)));
            }
        }

        let result = json!({
            "text": final_text,
            "steps": final_steps,
            "usage": {
                "input_tokens": final_input_tokens,
                "output_tokens": final_output_tokens,
            },
        });

        serde_json::to_string(&result).map_err(|e| py_err(format!("Serialize error: {e}")))
    })
}
