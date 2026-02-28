use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::error;
use crate::message::{Message, Usage};
use crate::provider::{FinishReason, GenerateOptions, Provider, ReasoningEffort};
use crate::tool::{Tool, ToolChoice};

/// Output from a complete agent run.
#[derive(Debug, Clone)]
pub struct AgentOutput {
    /// The final assistant message (text or structured).
    pub text: String,
    /// All messages in the conversation (including tool calls/results).
    pub messages: Vec<Message>,
    /// Total token usage across all steps.
    pub usage: Usage,
    /// Number of steps executed.
    pub steps: usize,
    /// Per-step results.
    pub step_results: Vec<StepResult>,
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
    provider: Arc<dyn Provider>,
    instructions: Option<String>,
    tools: Vec<Tool>,
    options: GenerateOptions,
    max_steps: usize,
}

impl Agent {
    pub fn builder(name: impl Into<String>, provider: Arc<dyn Provider>) -> AgentBuilder {
        AgentBuilder {
            name: name.into(),
            provider,
            instructions: None,
            tools: Vec::new(),
            options: GenerateOptions::default(),
            max_steps: 10,
        }
    }

    /// Run the agent with the given messages.
    pub async fn run(&self, messages: Vec<Message>) -> error::Result<AgentOutput> {
        let mut all_messages = Vec::new();
        let mut total_usage = Usage::default();
        let mut step_results = Vec::new();

        // Prepend system message if instructions are set
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

            // Accumulate usage
            total_usage.input_tokens += result.usage.input_tokens;
            total_usage.output_tokens += result.usage.output_tokens;
            if let Some(rt) = result.usage.reasoning_tokens {
                *total_usage.reasoning_tokens.get_or_insert(0) += rt;
            }

            let tool_calls_in_step = result.message.tool_calls();
            let has_tool_calls = !tool_calls_in_step.is_empty();

            // Collect tool call info
            let tool_call_infos: Vec<ToolCallInfo> = tool_calls_in_step
                .iter()
                .map(|(id, name, args)| ToolCallInfo {
                    id: id.to_string(),
                    name: name.to_string(),
                    arguments: (*args).clone(),
                })
                .collect();

            all_messages.push(result.message.clone());

            if !has_tool_calls || result.finish_reason != FinishReason::ToolCalls {
                // No tool calls — we're done
                step_results.push(StepResult {
                    step_index: step,
                    message: result.message,
                    finish_reason: result.finish_reason,
                    usage: result.usage,
                    tool_calls: tool_call_infos,
                    tool_results: Vec::new(),
                });
                break;
            }

            // Execute tool calls
            let mut tool_results = Vec::new();
            for (tc_id, tc_name, tc_args) in &tool_calls_in_step {
                debug!(tool = tc_name, "Executing tool");

                let tool = self.tools.iter().find(|t| t.name == *tc_name);
                match tool {
                    Some(t) => match t.execute((*tc_args).clone()).await {
                        Ok(result_val) => {
                            tool_results.push(ToolResultInfo {
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
                            tool_results.push(ToolResultInfo {
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
                        tool_results.push(ToolResultInfo {
                            tool_call_id: tc_id.to_string(),
                            tool_name: tc_name.to_string(),
                            result: error_val.clone(),
                            is_error: true,
                        });
                        all_messages.push(Message::tool_result(*tc_id, error_val));
                    }
                }
            }

            step_results.push(StepResult {
                step_index: step,
                message: result.message,
                finish_reason: result.finish_reason,
                usage: result.usage,
                tool_calls: tool_call_infos,
                tool_results,
            });
        }

        // Extract final text
        let final_text = all_messages
            .iter()
            .rev()
            .find(|m| m.role == crate::message::Role::Assistant)
            .and_then(|m| m.text())
            .unwrap_or("")
            .to_string();

        Ok(AgentOutput {
            text: final_text,
            messages: all_messages,
            usage: total_usage,
            steps: step_results.len(),
            step_results,
        })
    }
}

/// Builder for constructing Agent instances.
pub struct AgentBuilder {
    name: String,
    provider: Arc<dyn Provider>,
    instructions: Option<String>,
    tools: Vec<Tool>,
    options: GenerateOptions,
    max_steps: usize,
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
        }
    }
}
