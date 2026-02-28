use futures::Stream;
use std::pin::Pin;
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::error::{self, GaussError};
use crate::message::{Message, Usage};
use crate::provider::{FinishReason, GenerateOptions, Provider, ReasoningEffort};
use crate::streaming::StreamEvent;
use crate::tool::{Tool, ToolChoice};

/// Conditions that stop the agent loop.
#[derive(Debug, Clone)]
pub enum StopCondition {
    /// Stop after N steps regardless.
    MaxSteps(usize),
    /// Stop when a specific tool is called.
    HasToolCall(String),
    /// Stop when the model generates text (no tool calls).
    TextGenerated,
    /// Custom condition via callback name (evaluated externally).
    Custom(String),
}

/// Callback types for agent events.
#[cfg(not(target_arch = "wasm32"))]
pub type OnStepFinishFn =
    Arc<dyn Fn(&StepResult) -> Pin<Box<dyn std::future::Future<Output = ()> + Send>> + Send + Sync>;
#[cfg(not(target_arch = "wasm32"))]
pub type OnToolCallFn = Arc<
    dyn Fn(&ToolCallInfo) -> Pin<Box<dyn std::future::Future<Output = ()> + Send>> + Send + Sync,
>;

#[cfg(target_arch = "wasm32")]
pub type OnStepFinishFn =
    std::rc::Rc<dyn Fn(&StepResult) -> Pin<Box<dyn std::future::Future<Output = ()>>>>;
#[cfg(target_arch = "wasm32")]
pub type OnToolCallFn =
    std::rc::Rc<dyn Fn(&ToolCallInfo) -> Pin<Box<dyn std::future::Future<Output = ()>>>>;

/// Output from a complete agent run.
#[derive(Debug, Clone)]
pub struct AgentOutput {
    pub text: String,
    pub messages: Vec<Message>,
    pub usage: Usage,
    pub steps: usize,
    pub step_results: Vec<StepResult>,
    pub structured_output: Option<serde_json::Value>,
}

/// Result from a single agent step.
#[derive(Debug, Clone)]
pub struct StepResult {
    pub step_index: usize,
    pub message: Message,
    pub finish_reason: FinishReason,
    pub usage: Usage,
    pub tool_calls: Vec<ToolCallInfo>,
    pub tool_results: Vec<ToolResultInfo>,
}

#[derive(Debug, Clone)]
pub struct ToolCallInfo {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct ToolResultInfo {
    pub tool_call_id: String,
    pub tool_name: String,
    pub result: serde_json::Value,
    pub is_error: bool,
}

/// The main Agent. Replaces ToolLoopAgent from AI SDK.
pub struct Agent {
    pub name: String,
    provider: crate::Shared<dyn Provider>,
    instructions: Option<String>,
    tools: Vec<Tool>,
    options: GenerateOptions,
    max_steps: usize,
    stop_conditions: Vec<StopCondition>,
    on_step_finish: Option<OnStepFinishFn>,
    on_tool_call: Option<OnToolCallFn>,
}

impl Clone for Agent {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            provider: self.provider.clone(),
            instructions: self.instructions.clone(),
            tools: self.tools.clone(),
            options: self.options.clone(),
            max_steps: self.max_steps,
            stop_conditions: self.stop_conditions.clone(),
            on_step_finish: self.on_step_finish.clone(),
            on_tool_call: self.on_tool_call.clone(),
        }
    }
}

impl Agent {
    pub fn builder(name: impl Into<String>, provider: crate::Shared<dyn Provider>) -> AgentBuilder {
        AgentBuilder {
            name: name.into(),
            provider,
            instructions: None,
            tools: Vec::new(),
            options: GenerateOptions::default(),
            max_steps: 10,
            stop_conditions: Vec::new(),
            on_step_finish: None,
            on_tool_call: None,
        }
    }

    /// Check if any stop condition is met.
    fn should_stop(&self, step: &StepResult) -> bool {
        for cond in &self.stop_conditions {
            match cond {
                StopCondition::TextGenerated => {
                    if step.tool_calls.is_empty() {
                        return true;
                    }
                }
                StopCondition::HasToolCall(name) => {
                    if step.tool_calls.iter().any(|tc| tc.name == *name) {
                        return true;
                    }
                }
                StopCondition::MaxSteps(max) => {
                    if step.step_index + 1 >= *max {
                        return true;
                    }
                }
                StopCondition::Custom(_) => {
                    // Custom conditions are evaluated externally
                }
            }
        }
        false
    }

    /// Validate output against schema if provided.
    fn validate_output(&self, text: &str) -> error::Result<Option<serde_json::Value>> {
        if let Some(ref schema) = self.options.output_schema {
            let parsed: serde_json::Value =
                serde_json::from_str(text).map_err(|e| GaussError::SchemaValidation {
                    message: format!("Output is not valid JSON: {e}"),
                })?;

            // Use jsonschema for validation
            let validator =
                jsonschema::validator_for(schema).map_err(|e| GaussError::SchemaValidation {
                    message: format!("Invalid schema: {e}"),
                })?;

            let result = validator.validate(&parsed);
            if let Err(e) = result {
                return Err(GaussError::SchemaValidation {
                    message: format!("Schema validation failed: {e}"),
                });
            }

            Ok(Some(parsed))
        } else {
            Ok(None)
        }
    }

    /// Run the agent with the given messages.
    pub async fn run(&self, messages: Vec<Message>) -> error::Result<AgentOutput> {
        let mut all_messages = Vec::new();
        let mut total_usage = Usage::default();
        let mut step_results = Vec::new();

        if let Some(ref instructions) = self.instructions {
            all_messages.push(Message::system(instructions.clone()));
        }
        all_messages.extend(messages);

        for step in 0..self.max_steps {
            info!(agent = %self.name, step, "Executing step");

            let result = self
                .provider
                .generate(&all_messages, &self.tools, &self.options)
                .await?;

            total_usage.input_tokens += result.usage.input_tokens;
            total_usage.output_tokens += result.usage.output_tokens;
            if let Some(rt) = result.usage.reasoning_tokens {
                *total_usage.reasoning_tokens.get_or_insert(0) += rt;
            }

            let tool_calls_in_step = result.message.tool_calls();
            let has_tool_calls = !tool_calls_in_step.is_empty();

            let tool_call_infos: Vec<ToolCallInfo> = tool_calls_in_step
                .iter()
                .map(|(id, name, args)| ToolCallInfo {
                    id: id.to_string(),
                    name: name.to_string(),
                    arguments: (*args).clone(),
                })
                .collect();

            // Fire tool call callbacks
            if let Some(ref on_tool_call) = self.on_tool_call {
                for tc in &tool_call_infos {
                    on_tool_call(tc).await;
                }
            }

            all_messages.push(result.message.clone());

            if !has_tool_calls || result.finish_reason != FinishReason::ToolCalls {
                let step_result = StepResult {
                    step_index: step,
                    message: result.message,
                    finish_reason: result.finish_reason,
                    usage: result.usage,
                    tool_calls: tool_call_infos,
                    tool_results: Vec::new(),
                };

                if let Some(ref on_step_finish) = self.on_step_finish {
                    on_step_finish(&step_result).await;
                }

                step_results.push(step_result);
                break;
            }

            // Execute tool calls
            let mut tool_results_vec = Vec::new();
            for (tc_id, tc_name, tc_args) in &tool_calls_in_step {
                debug!(tool = tc_name, "Executing tool");

                let tool = self.tools.iter().find(|t| t.name == *tc_name);
                match tool {
                    Some(t) => match t.execute((*tc_args).clone()).await {
                        Ok(result_val) => {
                            tool_results_vec.push(ToolResultInfo {
                                tool_call_id: tc_id.to_string(),
                                tool_name: tc_name.to_string(),
                                result: result_val.clone(),
                                is_error: false,
                            });
                            all_messages.push(Message::tool_result(*tc_id, result_val));
                        }
                        Err(e) => {
                            warn!(tool = tc_name, error = %e, "Tool execution failed");
                            let error_val = serde_json::Value::String(format!("Error: {e}"));
                            tool_results_vec.push(ToolResultInfo {
                                tool_call_id: tc_id.to_string(),
                                tool_name: tc_name.to_string(),
                                result: error_val.clone(),
                                is_error: true,
                            });
                            all_messages.push(Message::tool_result(*tc_id, error_val));
                        }
                    },
                    None => {
                        warn!(tool = tc_name, "Tool not found");
                        let error_val =
                            serde_json::Value::String(format!("Error: Tool '{tc_name}' not found"));
                        tool_results_vec.push(ToolResultInfo {
                            tool_call_id: tc_id.to_string(),
                            tool_name: tc_name.to_string(),
                            result: error_val.clone(),
                            is_error: true,
                        });
                        all_messages.push(Message::tool_result(*tc_id, error_val));
                    }
                }
            }

            let step_result = StepResult {
                step_index: step,
                message: result.message,
                finish_reason: result.finish_reason,
                usage: result.usage,
                tool_calls: tool_call_infos,
                tool_results: tool_results_vec,
            };

            // Check stop conditions
            let should_stop = self.should_stop(&step_result);

            if let Some(ref on_step_finish) = self.on_step_finish {
                on_step_finish(&step_result).await;
            }

            step_results.push(step_result);

            if should_stop {
                break;
            }
        }

        let final_text = all_messages
            .iter()
            .rev()
            .find(|m| m.role == crate::message::Role::Assistant)
            .and_then(|m| m.text())
            .unwrap_or("")
            .to_string();

        // Validate structured output if schema is provided
        let structured_output = if !final_text.is_empty() {
            self.validate_output(&final_text).ok().flatten()
        } else {
            None
        };

        Ok(AgentOutput {
            text: final_text,
            messages: all_messages,
            usage: total_usage,
            steps: step_results.len(),
            step_results,
            structured_output,
        })
    }

    /// Stream agent execution, yielding events per step.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn run_stream(
        &self,
        messages: Vec<Message>,
    ) -> error::Result<Pin<Box<dyn Stream<Item = error::Result<AgentStreamEvent>> + Send + '_>>>
    {
        let mut all_messages = Vec::new();

        if let Some(ref instructions) = self.instructions {
            all_messages.push(Message::system(instructions.clone()));
        }
        all_messages.extend(messages);

        let stream = async_stream::stream! {
            for step in 0..self.max_steps {
                yield Ok(AgentStreamEvent::StepStart { step });

                let stream_result = self
                    .provider
                    .stream(&all_messages, &self.tools, &self.options)
                    .await;

                let mut inner_stream = match stream_result {
                    Ok(s) => s,
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                };

                use futures::StreamExt;
                let mut text_buffer = String::new();
                let mut tool_call_buffers: Vec<(String, String, String)> = Vec::new(); // (id, name, args)
                let mut step_finish_reason = FinishReason::Stop;
                let mut step_usage = Usage::default();

                while let Some(event) = inner_stream.next().await {
                    match event {
                        Ok(StreamEvent::TextDelta(delta)) => {
                            text_buffer.push_str(&delta);
                            yield Ok(AgentStreamEvent::TextDelta { step, delta });
                        }
                        Ok(StreamEvent::ToolCallDelta { index, id, name, arguments_delta }) => {
                            while tool_call_buffers.len() <= index {
                                tool_call_buffers.push((String::new(), String::new(), String::new()));
                            }
                            if let Some(id) = id {
                                tool_call_buffers[index].0 = id;
                            }
                            if let Some(name) = name {
                                tool_call_buffers[index].1 = name;
                            }
                            if let Some(args) = arguments_delta {
                                tool_call_buffers[index].2.push_str(&args);
                            }
                            yield Ok(AgentStreamEvent::ToolCallDelta { step, index });
                        }
                        Ok(StreamEvent::FinishReason(fr)) => {
                            step_finish_reason = fr;
                        }
                        Ok(StreamEvent::Usage(u)) => {
                            step_usage = u;
                        }
                        Ok(StreamEvent::Done) => break,
                        Ok(other) => {
                            yield Ok(AgentStreamEvent::RawEvent { step, event: other });
                        }
                        Err(e) => {
                            yield Err(e);
                            return;
                        }
                    }
                }

                let has_tool_calls = !tool_call_buffers.is_empty()
                    && tool_call_buffers.iter().any(|(_, name, _)| !name.is_empty());

                yield Ok(AgentStreamEvent::StepFinish {
                    step,
                    finish_reason: step_finish_reason.clone(),
                    has_tool_calls,
                });

                if !has_tool_calls || step_finish_reason != FinishReason::ToolCalls {
                    yield Ok(AgentStreamEvent::Done {
                        text: text_buffer,
                        steps: step + 1,
                        usage: step_usage,
                    });
                    return;
                }

                // Build assistant message with tool calls and execute them
                let mut assistant_content = Vec::new();
                if !text_buffer.is_empty() {
                    assistant_content.push(crate::message::Content::Text { text: text_buffer.clone() });
                }
                for (id, name, args_str) in &tool_call_buffers {
                    if !name.is_empty() {
                        let arguments = serde_json::from_str(args_str).unwrap_or(serde_json::json!({}));
                        assistant_content.push(crate::message::Content::ToolCall {
                            id: id.clone(),
                            name: name.clone(),
                            arguments,
                        });
                    }
                }
                all_messages.push(Message {
                    role: crate::message::Role::Assistant,
                    content: assistant_content,
                    name: None,
                });

                // Execute tool calls
                for (tc_id, tc_name, tc_args_str) in &tool_call_buffers {
                    if tc_name.is_empty() { continue; }
                    let tc_args: serde_json::Value = serde_json::from_str(tc_args_str).unwrap_or(serde_json::json!({}));

                    let tool = self.tools.iter().find(|t| t.name == *tc_name);
                    match tool {
                        Some(t) => match t.execute(tc_args).await {
                            Ok(result_val) => {
                                yield Ok(AgentStreamEvent::ToolResult {
                                    step,
                                    tool_name: tc_name.clone(),
                                    result: result_val.clone(),
                                    is_error: false,
                                });
                                all_messages.push(Message::tool_result(tc_id.as_str(), result_val));
                            }
                            Err(e) => {
                                let error_val = serde_json::Value::String(format!("Error: {e}"));
                                yield Ok(AgentStreamEvent::ToolResult {
                                    step,
                                    tool_name: tc_name.clone(),
                                    result: error_val.clone(),
                                    is_error: true,
                                });
                                all_messages.push(Message::tool_result(tc_id.as_str(), error_val));
                            }
                        },
                        None => {
                            let error_val = serde_json::Value::String(format!("Error: Tool '{tc_name}' not found"));
                            yield Ok(AgentStreamEvent::ToolResult {
                                step,
                                tool_name: tc_name.clone(),
                                result: error_val.clone(),
                                is_error: true,
                            });
                            all_messages.push(Message::tool_result(tc_id.as_str(), error_val));
                        }
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }

    /// Stream agent execution (WASM — no Send bound).
    #[cfg(target_arch = "wasm32")]
    pub async fn run_stream(
        &self,
        messages: Vec<Message>,
    ) -> error::Result<Pin<Box<dyn Stream<Item = error::Result<AgentStreamEvent>> + '_>>>
    {
        // Delegate to the same internal logic via run() for WASM
        // Full streaming support on WASM requires further async refactoring
        let output = self.run(messages).await?;
        let events: Vec<error::Result<AgentStreamEvent>> = vec![
            Ok(AgentStreamEvent::StepStart { step: 0 }),
            Ok(AgentStreamEvent::StepFinish {
                step: 0,
                finish_reason: crate::provider::FinishReason::Stop,
                has_tool_calls: false,
            }),
        ];
        Ok(Box::pin(futures::stream::iter(events)))
    }
}

/// Events emitted during agent streaming.
#[derive(Debug, Clone)]
pub enum AgentStreamEvent {
    StepStart {
        step: usize,
    },
    TextDelta {
        step: usize,
        delta: String,
    },
    ToolCallDelta {
        step: usize,
        index: usize,
    },
    ToolResult {
        step: usize,
        tool_name: String,
        result: serde_json::Value,
        is_error: bool,
    },
    StepFinish {
        step: usize,
        finish_reason: FinishReason,
        has_tool_calls: bool,
    },
    RawEvent {
        step: usize,
        event: StreamEvent,
    },
    Done {
        text: String,
        steps: usize,
        usage: Usage,
    },
}

/// Builder for constructing Agent instances.
pub struct AgentBuilder {
    name: String,
    provider: crate::Shared<dyn Provider>,
    instructions: Option<String>,
    tools: Vec<Tool>,
    options: GenerateOptions,
    max_steps: usize,
    stop_conditions: Vec<StopCondition>,
    on_step_finish: Option<OnStepFinishFn>,
    on_tool_call: Option<OnToolCallFn>,
}

impl AgentBuilder {
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    pub fn tool(mut self, tool: Tool) -> Self {
        self.tools.push(tool);
        self
    }

    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = tools;
        self
    }

    pub fn max_steps(mut self, steps: usize) -> Self {
        self.max_steps = steps;
        self
    }

    pub fn stop_when(mut self, condition: StopCondition) -> Self {
        self.stop_conditions.push(condition);
        self
    }

    pub fn on_step_finish(mut self, callback: OnStepFinishFn) -> Self {
        self.on_step_finish = Some(callback);
        self
    }

    pub fn on_tool_call(mut self, callback: OnToolCallFn) -> Self {
        self.on_tool_call = Some(callback);
        self
    }

    pub fn temperature(mut self, temp: f64) -> Self {
        self.options.temperature = Some(temp);
        self
    }

    pub fn top_p(mut self, top_p: f64) -> Self {
        self.options.top_p = Some(top_p);
        self
    }

    pub fn top_k(mut self, top_k: u32) -> Self {
        self.options.top_k = Some(top_k);
        self
    }

    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.options.max_tokens = Some(tokens);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.options.seed = Some(seed);
        self
    }

    pub fn reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.options.reasoning_effort = Some(effort);
        self
    }

    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.options.tool_choice = Some(choice);
        self
    }

    pub fn frequency_penalty(mut self, penalty: f64) -> Self {
        self.options.frequency_penalty = Some(penalty);
        self
    }

    pub fn presence_penalty(mut self, penalty: f64) -> Self {
        self.options.presence_penalty = Some(penalty);
        self
    }

    pub fn output_schema(mut self, schema: serde_json::Value) -> Self {
        self.options.output_schema = Some(schema);
        self
    }

    pub fn build(self) -> Agent {
        Agent {
            name: self.name,
            provider: self.provider,
            instructions: self.instructions,
            tools: self.tools,
            options: self.options,
            max_steps: self.max_steps,
            stop_conditions: self.stop_conditions,
            on_step_finish: self.on_step_finish,
            on_tool_call: self.on_tool_call,
        }
    }
}
