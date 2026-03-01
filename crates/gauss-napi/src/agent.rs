use crate::provider::get_provider;
use crate::types::*;
use gauss_core::agent::{Agent as RustAgent, AgentStreamEvent, StopCondition};
use gauss_core::message::Message as RustMessage;
use gauss_core::tool::Tool as RustTool;
use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ErrorStrategy, ThreadsafeFunction};
use serde_json::json;
use std::sync::Arc;

fn build_agent(
    name: String,
    provider: Arc<dyn gauss_core::provider::Provider>,
    tools: &[ToolDef],
    opts: AgentOptions,
) -> RustAgent {
    let mut builder = RustAgent::builder(name, provider);

    if let Some(instructions) = opts.instructions {
        builder = builder.instructions(instructions);
    }
    if let Some(max_steps) = opts.max_steps {
        builder = builder.max_steps(max_steps as usize);
    }
    if let Some(temp) = opts.temperature {
        builder = builder.temperature(temp);
    }
    if let Some(tp) = opts.top_p {
        builder = builder.top_p(tp);
    }
    if let Some(mt) = opts.max_tokens {
        builder = builder.max_tokens(mt);
    }
    if let Some(seed) = opts.seed {
        builder = builder.seed(seed as u64);
    }
    if let Some(ref schema) = opts.output_schema {
        builder = builder.output_schema(schema.clone());
    }
    if let Some(tool_name) = opts.stop_on_tool {
        builder = builder.stop_when(StopCondition::HasToolCall(tool_name));
    }
    if let Some(budget) = opts.thinking_budget {
        builder = builder.thinking_budget(budget);
    }
    if let Some(ref effort_str) = opts.reasoning_effort {
        if let Some(effort) = parse_reasoning_effort(effort_str) {
            builder = builder.reasoning_effort(effort);
        }
    }
    if let Some(true) = opts.cache_control {
        builder = builder.cache_control(true);
    }
    if let Some(ref ce) = opts.code_execution {
        builder = apply_code_execution(builder, ce);
    }
    if let Some(true) = opts.grounding {
        builder = builder.grounding(true);
    }
    if let Some(true) = opts.native_code_execution {
        builder = builder.native_code_execution(true);
    }
    if let Some(ref modalities) = opts.response_modalities {
        builder = builder.response_modalities(modalities.clone());
    }

    for td in tools {
        let mut tool_builder = RustTool::builder(&td.name, &td.description);
        if let Some(ref params) = td.parameters {
            tool_builder = tool_builder.parameters_json(params.clone());
        }
        builder = builder.tool(tool_builder.build());
    }

    builder.build()
}

/// Run an agent with the given provider, tools, messages, and options.
#[napi]
pub async fn agent_run(
    name: String,
    provider_handle: u32,
    tools: Vec<ToolDef>,
    messages: Vec<JsMessage>,
    options: Option<AgentOptions>,
) -> Result<AgentResult> {
    let provider = get_provider(provider_handle)?;
    let opts = options.unwrap_or_default();
    let agent = build_agent(name, provider, &tools, opts);
    let rust_messages: Vec<RustMessage> = messages.iter().map(js_message_to_rust).collect();

    let output = agent
        .run(rust_messages)
        .await
        .map_err(|e| napi::Error::from_reason(format!("Agent error: {e}")))?;

    Ok(rust_output_to_js(output))
}

/// Run an agent where tool execution is delegated back to JavaScript.
#[napi]
pub async fn agent_run_with_tool_executor(
    name: String,
    provider_handle: u32,
    tools: Vec<ToolDef>,
    messages: Vec<JsMessage>,
    options: Option<AgentOptions>,
    #[napi(ts_arg_type = "(callJson: string) => Promise<string>")]
    tool_executor: ThreadsafeFunction<String, ErrorStrategy::Fatal>,
) -> Result<AgentResult> {
    let provider = get_provider(provider_handle)?;
    let opts = options.unwrap_or_default();

    let mut builder = RustAgent::builder(name, provider);
    // Apply options manually since we need custom tool executors
    if let Some(instructions) = opts.instructions {
        builder = builder.instructions(instructions);
    }
    if let Some(max_steps) = opts.max_steps {
        builder = builder.max_steps(max_steps as usize);
    }
    if let Some(temp) = opts.temperature {
        builder = builder.temperature(temp);
    }
    if let Some(tp) = opts.top_p {
        builder = builder.top_p(tp);
    }
    if let Some(mt) = opts.max_tokens {
        builder = builder.max_tokens(mt);
    }
    if let Some(seed) = opts.seed {
        builder = builder.seed(seed as u64);
    }
    if let Some(ref schema) = opts.output_schema {
        builder = builder.output_schema(schema.clone());
    }
    if let Some(tool_name) = opts.stop_on_tool {
        builder = builder.stop_when(StopCondition::HasToolCall(tool_name));
    }
    if let Some(budget) = opts.thinking_budget {
        builder = builder.thinking_budget(budget);
    }
    if let Some(ref effort_str) = opts.reasoning_effort {
        if let Some(effort) = parse_reasoning_effort(effort_str) {
            builder = builder.reasoning_effort(effort);
        }
    }
    if let Some(true) = opts.cache_control {
        builder = builder.cache_control(true);
    }
    if let Some(ref ce) = opts.code_execution {
        builder = apply_code_execution(builder, ce);
    }
    if let Some(true) = opts.grounding {
        builder = builder.grounding(true);
    }
    if let Some(true) = opts.native_code_execution {
        builder = builder.native_code_execution(true);
    }
    if let Some(ref modalities) = opts.response_modalities {
        builder = builder.response_modalities(modalities.clone());
    }

    let tool_executor = Arc::new(tool_executor);

    for td in &tools {
        let tool_exec = tool_executor.clone();
        let tool_name = td.name.clone();

        let mut tool_builder = RustTool::builder(&td.name, &td.description);
        if let Some(ref params) = td.parameters {
            tool_builder = tool_builder.parameters_json(params.clone());
        }

        tool_builder = tool_builder.execute(move |args: serde_json::Value| {
            let exec = tool_exec.clone();
            let tn = tool_name.clone();
            async move {
                let call_json = serde_json::to_string(&json!({
                    "tool": tn,
                    "args": args
                }))
                .map_err(|e| gauss_core::error::GaussError::tool(&tn, format!("Serialize: {e}")))?;

                let promise: Promise<String> = exec
                    .call_async::<Promise<String>>(call_json)
                    .await
                    .map_err(|e| {
                        gauss_core::error::GaussError::tool(&tn, format!("NAPI call error: {e}"))
                    })?;

                let result_str = promise.await.map_err(|e| {
                    gauss_core::error::GaussError::tool(&tn, format!("JS Promise error: {e}"))
                })?;

                serde_json::from_str(&result_str).map_err(|e| {
                    gauss_core::error::GaussError::tool(&tn, format!("Deserialize: {e}"))
                })
            }
        });

        builder = builder.tool(tool_builder.build());
    }

    let agent = builder.build();
    let rust_messages: Vec<RustMessage> = messages.iter().map(js_message_to_rust).collect();

    let output = agent
        .run(rust_messages)
        .await
        .map_err(|e| napi::Error::from_reason(format!("Agent error: {e}")))?;

    Ok(rust_output_to_js(output))
}

/// Stream an agent execution, pushing events to JS via callback.
#[napi]
pub async fn agent_stream_with_tool_executor(
    name: String,
    provider_handle: u32,
    tools: Vec<ToolDef>,
    messages: Vec<JsMessage>,
    options: Option<AgentOptions>,
    #[napi(ts_arg_type = "(eventJson: string) => void")] stream_callback: ThreadsafeFunction<
        String,
        ErrorStrategy::Fatal,
    >,
    #[napi(ts_arg_type = "(callJson: string) => Promise<string>")]
    tool_executor: ThreadsafeFunction<String, ErrorStrategy::Fatal>,
) -> Result<AgentResult> {
    use futures::StreamExt;

    let provider = get_provider(provider_handle)?;
    let opts = options.unwrap_or_default();

    let mut builder = RustAgent::builder(name, provider);
    if let Some(instructions) = opts.instructions {
        builder = builder.instructions(instructions);
    }
    if let Some(max_steps) = opts.max_steps {
        builder = builder.max_steps(max_steps as usize);
    }
    if let Some(temp) = opts.temperature {
        builder = builder.temperature(temp);
    }
    if let Some(tp) = opts.top_p {
        builder = builder.top_p(tp);
    }
    if let Some(mt) = opts.max_tokens {
        builder = builder.max_tokens(mt);
    }
    if let Some(seed) = opts.seed {
        builder = builder.seed(seed as u64);
    }
    if let Some(ref schema) = opts.output_schema {
        builder = builder.output_schema(schema.clone());
    }
    if let Some(tool_name) = opts.stop_on_tool {
        builder = builder.stop_when(StopCondition::HasToolCall(tool_name));
    }
    if let Some(budget) = opts.thinking_budget {
        builder = builder.thinking_budget(budget);
    }
    if let Some(ref effort_str) = opts.reasoning_effort {
        if let Some(effort) = parse_reasoning_effort(effort_str) {
            builder = builder.reasoning_effort(effort);
        }
    }
    if let Some(true) = opts.cache_control {
        builder = builder.cache_control(true);
    }
    if let Some(ref ce) = opts.code_execution {
        builder = apply_code_execution(builder, ce);
    }
    if let Some(true) = opts.grounding {
        builder = builder.grounding(true);
    }
    if let Some(true) = opts.native_code_execution {
        builder = builder.native_code_execution(true);
    }
    if let Some(ref modalities) = opts.response_modalities {
        builder = builder.response_modalities(modalities.clone());
    }

    let tool_executor = Arc::new(tool_executor);

    for td in &tools {
        let tool_exec = tool_executor.clone();
        let tool_name = td.name.clone();

        let mut tool_builder = RustTool::builder(&td.name, &td.description);
        if let Some(ref params) = td.parameters {
            tool_builder = tool_builder.parameters_json(params.clone());
        }

        tool_builder = tool_builder.execute(move |args: serde_json::Value| {
            let exec = tool_exec.clone();
            let tn = tool_name.clone();
            async move {
                let call_json = serde_json::to_string(&json!({
                    "tool": tn,
                    "args": args
                }))
                .map_err(|e| gauss_core::error::GaussError::tool(&tn, format!("Serialize: {e}")))?;

                let promise: Promise<String> = exec
                    .call_async::<Promise<String>>(call_json)
                    .await
                    .map_err(|e| {
                        gauss_core::error::GaussError::tool(&tn, format!("NAPI call error: {e}"))
                    })?;

                let result_str = promise.await.map_err(|e| {
                    gauss_core::error::GaussError::tool(&tn, format!("JS Promise error: {e}"))
                })?;

                serde_json::from_str(&result_str).map_err(|e| {
                    gauss_core::error::GaussError::tool(&tn, format!("Deserialize: {e}"))
                })
            }
        });

        builder = builder.tool(tool_builder.build());
    }

    let agent = builder.build();
    let rust_messages: Vec<RustMessage> = messages.iter().map(js_message_to_rust).collect();

    let mut stream = agent
        .run_stream(rust_messages)
        .await
        .map_err(|e| napi::Error::from_reason(format!("Stream init error: {e}")))?;

    let mut final_text = String::new();
    let mut final_steps = 0u32;
    let mut final_input_tokens = 0u32;
    let mut final_output_tokens = 0u32;

    while let Some(event) = stream.next().await {
        match event {
            Ok(AgentStreamEvent::TextDelta { step, delta }) => {
                let event_json = serde_json::to_string(&json!({
                    "type": "text_delta",
                    "step": step,
                    "delta": delta,
                }))
                .unwrap_or_default();
                let _ = stream_callback.call(
                    event_json,
                    napi::threadsafe_function::ThreadsafeFunctionCallMode::NonBlocking,
                );
            }
            Ok(AgentStreamEvent::StepStart { step }) => {
                let event_json = serde_json::to_string(&json!({
                    "type": "step_start",
                    "step": step,
                }))
                .unwrap_or_default();
                let _ = stream_callback.call(
                    event_json,
                    napi::threadsafe_function::ThreadsafeFunctionCallMode::NonBlocking,
                );
            }
            Ok(AgentStreamEvent::ToolCallDelta { step, index }) => {
                let event_json = serde_json::to_string(&json!({
                    "type": "tool_call_delta",
                    "step": step,
                    "index": index,
                }))
                .unwrap_or_default();
                let _ = stream_callback.call(
                    event_json,
                    napi::threadsafe_function::ThreadsafeFunctionCallMode::NonBlocking,
                );
            }
            Ok(AgentStreamEvent::ToolResult {
                step,
                tool_name,
                result,
                is_error,
            }) => {
                let event_json = serde_json::to_string(&json!({
                    "type": "tool_result",
                    "step": step,
                    "toolName": tool_name,
                    "result": result,
                    "isError": is_error,
                }))
                .unwrap_or_default();
                let _ = stream_callback.call(
                    event_json,
                    napi::threadsafe_function::ThreadsafeFunctionCallMode::NonBlocking,
                );
            }
            Ok(AgentStreamEvent::StepFinish {
                step,
                finish_reason,
                has_tool_calls,
            }) => {
                let event_json = serde_json::to_string(&json!({
                    "type": "step_finish",
                    "step": step,
                    "finishReason": format!("{:?}", finish_reason),
                    "hasToolCalls": has_tool_calls,
                }))
                .unwrap_or_default();
                let _ = stream_callback.call(
                    event_json,
                    napi::threadsafe_function::ThreadsafeFunctionCallMode::NonBlocking,
                );
            }
            Ok(AgentStreamEvent::Done { text, steps, usage }) => {
                final_text = text;
                final_steps = steps as u32;
                final_input_tokens = usage.input_tokens as u32;
                final_output_tokens = usage.output_tokens as u32;

                let event_json = serde_json::to_string(&json!({
                    "type": "done",
                    "text": final_text,
                    "steps": final_steps,
                    "inputTokens": final_input_tokens,
                    "outputTokens": final_output_tokens,
                }))
                .unwrap_or_default();
                let _ = stream_callback.call(
                    event_json,
                    napi::threadsafe_function::ThreadsafeFunctionCallMode::NonBlocking,
                );
            }
            Ok(AgentStreamEvent::RawEvent { step, event }) => {
                let event_json = serde_json::to_string(&json!({
                    "type": "raw_event",
                    "step": step,
                    "event": format!("{:?}", event),
                }))
                .unwrap_or_default();
                let _ = stream_callback.call(
                    event_json,
                    napi::threadsafe_function::ThreadsafeFunctionCallMode::NonBlocking,
                );
            }
            Err(e) => {
                let event_json = serde_json::to_string(&json!({
                    "type": "error",
                    "error": format!("{e}"),
                }))
                .unwrap_or_default();
                let _ = stream_callback.call(
                    event_json,
                    napi::threadsafe_function::ThreadsafeFunctionCallMode::NonBlocking,
                );
                return Err(napi::Error::from_reason(format!("Stream error: {e}")));
            }
        }
    }

    Ok(AgentResult {
        text: final_text,
        steps: final_steps,
        input_tokens: final_input_tokens,
        output_tokens: final_output_tokens,
        structured_output: None,
        thinking: None,
        citations: vec![],
        grounding_metadata: None,
    })
}
