//! Advanced agent patterns: reflection, planning, tool composition, validation.

use serde::{Deserialize, Serialize};

use crate::agent::{Agent, AgentOutput};
use crate::error::{self, GaussError};
use crate::message::Message;
use crate::tool::Tool;

// ---------------------------------------------------------------------------
// Tool Validation
// ---------------------------------------------------------------------------

/// Strategy for coercing invalid tool inputs before execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoercionStrategy {
    /// Replace null / undefined values with schema defaults.
    NullToDefault,
    /// Attempt to cast types (string→number, string→bool).
    TypeCast,
    /// Try parsing stringified JSON.
    JsonParse,
    /// Strip null fields entirely.
    StripNull,
}

/// Multi-stage validator that recovers from common LLM output quirks.
#[derive(Debug, Clone)]
pub struct ToolValidator {
    strategies: Vec<CoercionStrategy>,
}

impl Default for ToolValidator {
    fn default() -> Self {
        Self {
            strategies: vec![
                CoercionStrategy::NullToDefault,
                CoercionStrategy::TypeCast,
                CoercionStrategy::JsonParse,
                CoercionStrategy::StripNull,
            ],
        }
    }
}

impl ToolValidator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_strategies(strategies: Vec<CoercionStrategy>) -> Self {
        Self { strategies }
    }

    /// Apply all coercion strategies in order and validate against schema.
    pub fn validate(
        &self,
        mut input: serde_json::Value,
        schema: &serde_json::Value,
    ) -> error::Result<serde_json::Value> {
        for strategy in &self.strategies {
            input = match strategy {
                CoercionStrategy::NullToDefault => Self::apply_null_to_default(input, schema),
                CoercionStrategy::TypeCast => Self::apply_type_cast(input, schema),
                CoercionStrategy::JsonParse => Self::apply_json_parse(input),
                CoercionStrategy::StripNull => Self::apply_strip_null(input),
            };
        }

        // Final validation against schema
        if let Ok(validator) = jsonschema::validator_for(schema)
            && let Err(e) = validator.validate(&input)
        {
            return Err(GaussError::SchemaValidation {
                message: format!("Tool input validation failed after coercion: {e}"),
            });
        }

        Ok(input)
    }

    fn apply_null_to_default(
        input: serde_json::Value,
        schema: &serde_json::Value,
    ) -> serde_json::Value {
        let props = match schema.get("properties") {
            Some(p) => p,
            None => return input,
        };
        let mut obj = match input {
            serde_json::Value::Object(o) => o,
            other => return other,
        };
        if let Some(props_map) = props.as_object() {
            for (key, prop_schema) in props_map {
                let val = obj.get(key);
                if (val.is_none() || val == Some(&serde_json::Value::Null))
                    && let Some(default) = prop_schema.get("default")
                {
                    obj.insert(key.clone(), default.clone());
                }
            }
        }
        serde_json::Value::Object(obj)
    }

    fn apply_type_cast(input: serde_json::Value, schema: &serde_json::Value) -> serde_json::Value {
        let props = match schema.get("properties") {
            Some(p) => p,
            None => return input,
        };
        let mut obj = match input {
            serde_json::Value::Object(o) => o,
            other => return other,
        };
        if let Some(props_map) = props.as_object() {
            for (key, prop_schema) in props_map {
                if let Some(val) = obj.get(key).cloned()
                    && let Some(expected_type) = prop_schema.get("type").and_then(|t| t.as_str())
                {
                    let casted = Self::cast_value(val, expected_type);
                    obj.insert(key.clone(), casted);
                }
            }
        }
        serde_json::Value::Object(obj)
    }

    fn cast_value(val: serde_json::Value, expected_type: &str) -> serde_json::Value {
        match (expected_type, &val) {
            ("number" | "integer", serde_json::Value::String(s)) => {
                if let Ok(n) = s.parse::<f64>() {
                    if expected_type == "integer" {
                        serde_json::Value::Number(
                            serde_json::Number::from_f64(n.round())
                                .unwrap_or_else(|| serde_json::Number::from(0)),
                        )
                    } else {
                        serde_json::Value::Number(
                            serde_json::Number::from_f64(n)
                                .unwrap_or_else(|| serde_json::Number::from(0)),
                        )
                    }
                } else {
                    val
                }
            }
            ("boolean", serde_json::Value::String(s)) => match s.to_lowercase().as_str() {
                "true" | "1" | "yes" => serde_json::Value::Bool(true),
                "false" | "0" | "no" => serde_json::Value::Bool(false),
                _ => val,
            },
            ("string", serde_json::Value::Number(n)) => serde_json::Value::String(n.to_string()),
            ("string", serde_json::Value::Bool(b)) => serde_json::Value::String(b.to_string()),
            _ => val,
        }
    }

    fn apply_json_parse(input: serde_json::Value) -> serde_json::Value {
        if let serde_json::Value::Object(mut obj) = input {
            let keys: Vec<String> = obj.keys().cloned().collect();
            for key in keys {
                if let Some(serde_json::Value::String(s)) = obj.get(&key)
                    && ((s.starts_with('{') && s.ends_with('}'))
                        || (s.starts_with('[') && s.ends_with(']')))
                    && let Ok(parsed) = serde_json::from_str::<serde_json::Value>(s)
                {
                    obj.insert(key, parsed);
                }
            }
            serde_json::Value::Object(obj)
        } else {
            input
        }
    }

    fn apply_strip_null(input: serde_json::Value) -> serde_json::Value {
        if let serde_json::Value::Object(obj) = input {
            let filtered: serde_json::Map<String, serde_json::Value> =
                obj.into_iter().filter(|(_, v)| !v.is_null()).collect();
            serde_json::Value::Object(filtered)
        } else {
            input
        }
    }
}

// ---------------------------------------------------------------------------
// ToolChain
// ---------------------------------------------------------------------------

/// A chain of tools executed in sequence, piping output to the next input.
#[derive(Debug, Clone)]
pub struct ToolChain {
    name: String,
    tools: Vec<Tool>,
}

impl ToolChain {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            tools: Vec::new(),
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, tool: Tool) -> Self {
        self.tools.push(tool);
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Execute all tools in sequence. Each tool's output is passed as input to the next.
    pub async fn execute(
        &self,
        initial_input: serde_json::Value,
    ) -> error::Result<serde_json::Value> {
        let mut current = initial_input;
        for tool in &self.tools {
            current = tool.execute(current).await?;
        }
        Ok(current)
    }

    /// Execute and return intermediate results from each step.
    pub async fn execute_traced(
        &self,
        initial_input: serde_json::Value,
    ) -> error::Result<Vec<ToolChainStep>> {
        let mut steps = Vec::new();
        let mut current = initial_input;
        for tool in &self.tools {
            let input = current.clone();
            let output = tool.execute(current).await?;
            steps.push(ToolChainStep {
                tool_name: tool.name.clone(),
                input,
                output: output.clone(),
            });
            current = output;
        }
        Ok(steps)
    }
}

/// A single step in a tool chain execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChainStep {
    pub tool_name: String,
    pub input: serde_json::Value,
    pub output: serde_json::Value,
}

// ---------------------------------------------------------------------------
// ReflectionAgent
// ---------------------------------------------------------------------------

/// Configuration for the reflection loop.
#[derive(Debug, Clone)]
pub struct ReflectionConfig {
    /// Maximum number of reflection iterations.
    pub max_iterations: usize,
    /// System prompt for the critic agent.
    pub critic_instructions: String,
    /// If true, stop early when the critic approves.
    pub stop_on_approval: bool,
    /// Keyword in critic response that signals approval.
    pub approval_keyword: String,
}

impl Default for ReflectionConfig {
    fn default() -> Self {
        Self {
            max_iterations: 3,
            critic_instructions: "You are a critical reviewer. Analyze the following response for accuracy, completeness, and quality. If the response is satisfactory, respond with APPROVED. Otherwise, provide specific improvement suggestions.".to_string(),
            stop_on_approval: true,
            approval_keyword: "APPROVED".to_string(),
        }
    }
}

/// An agent that reflects on its own output and iteratively refines it.
pub struct ReflectionAgent {
    /// The primary agent that generates responses.
    agent: Agent,
    /// Configuration for the reflection loop.
    config: ReflectionConfig,
}

/// Result from a reflection loop.
#[derive(Debug, Clone)]
pub struct ReflectionResult {
    /// The final refined output.
    pub final_output: AgentOutput,
    /// History of all iterations (generation, critique pairs).
    pub iterations: Vec<ReflectionIteration>,
    /// Whether the critic approved the final output.
    pub approved: bool,
}

#[derive(Debug, Clone)]
pub struct ReflectionIteration {
    pub iteration: usize,
    pub generation: String,
    pub critique: String,
    pub approved: bool,
}

impl ReflectionAgent {
    pub fn new(agent: Agent, config: ReflectionConfig) -> Self {
        Self { agent, config }
    }

    pub fn with_defaults(agent: Agent) -> Self {
        Self::new(agent, ReflectionConfig::default())
    }

    /// Run the reflection loop.
    pub async fn run(&self, messages: Vec<Message>) -> error::Result<ReflectionResult> {
        let mut iterations = Vec::new();
        let mut last_output = self.agent.run(messages.clone()).await?;
        let mut approved = false;

        for i in 0..self.config.max_iterations {
            // Build critique prompt
            let critique_messages = vec![
                Message::system(self.config.critic_instructions.clone()),
                Message::user(format!(
                    "Please review this response:\n\n{}",
                    last_output.text
                )),
            ];

            let critique_output = self.agent.run(critique_messages).await?;
            let critique_text = critique_output.text.clone();
            let is_approved = critique_text.contains(&self.config.approval_keyword);

            iterations.push(ReflectionIteration {
                iteration: i,
                generation: last_output.text.clone(),
                critique: critique_text.clone(),
                approved: is_approved,
            });

            if is_approved && self.config.stop_on_approval {
                approved = true;
                break;
            }

            // Refine: send original + critique to agent
            let mut refine_messages = messages.clone();
            refine_messages.push(Message::assistant(last_output.text.clone()));
            refine_messages.push(Message::user(format!(
                "Based on this feedback, please improve your response:\n\n{}",
                critique_text
            )));

            last_output = self.agent.run(refine_messages).await?;
        }

        Ok(ReflectionResult {
            final_output: last_output,
            iterations,
            approved,
        })
    }
}

// ---------------------------------------------------------------------------
// PlanningAgent
// ---------------------------------------------------------------------------

/// A step in a plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    pub id: String,
    pub description: String,
    pub depends_on: Vec<String>,
    #[serde(default)]
    pub status: PlanStepStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanStepStatus {
    #[default]
    Pending,
    InProgress,
    Done,
    Failed,
}

/// A plan: ordered list of steps with dependencies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    pub goal: String,
    pub steps: Vec<PlanStep>,
}

impl Plan {
    /// Get steps that are ready (all dependencies done).
    pub fn ready_steps(&self) -> Vec<&PlanStep> {
        self.steps
            .iter()
            .filter(|s| {
                s.status == PlanStepStatus::Pending
                    && s.depends_on.iter().all(|dep| {
                        self.steps
                            .iter()
                            .any(|d| d.id == *dep && d.status == PlanStepStatus::Done)
                    })
            })
            .collect()
    }

    /// Check if all steps are done.
    pub fn is_complete(&self) -> bool {
        self.steps.iter().all(|s| s.status == PlanStepStatus::Done)
    }

    /// Check if any step has failed.
    pub fn has_failures(&self) -> bool {
        self.steps
            .iter()
            .any(|s| s.status == PlanStepStatus::Failed)
    }

    /// Mark a step's status.
    pub fn set_status(&mut self, step_id: &str, status: PlanStepStatus) {
        if let Some(step) = self.steps.iter_mut().find(|s| s.id == step_id) {
            step.status = status;
        }
    }

    /// Set a step's result.
    pub fn set_result(&mut self, step_id: &str, result: String) {
        if let Some(step) = self.steps.iter_mut().find(|s| s.id == step_id) {
            step.result = Some(result);
        }
    }
}

/// An agent that decomposes a task into a plan and executes it step by step.
pub struct PlanningAgent {
    agent: Agent,
    max_plan_steps: usize,
}

/// Result from a planning agent run.
#[derive(Debug, Clone)]
pub struct PlanningResult {
    pub plan: Plan,
    pub final_output: AgentOutput,
}

impl PlanningAgent {
    pub fn new(agent: Agent) -> Self {
        Self {
            agent,
            max_plan_steps: 10,
        }
    }

    pub fn max_plan_steps(mut self, max: usize) -> Self {
        self.max_plan_steps = max;
        self
    }

    /// Parse a plan from LLM JSON output.
    pub fn parse_plan(text: &str) -> error::Result<Plan> {
        // Try to extract JSON from the text (LLMs often wrap in markdown)
        let json_str = if let Some(start) = text.find('{') {
            if let Some(end) = text.rfind('}') {
                &text[start..=end]
            } else {
                text
            }
        } else {
            text
        };

        serde_json::from_str(json_str).map_err(|e| GaussError::SchemaValidation {
            message: format!("Failed to parse plan: {e}"),
        })
    }

    /// Run: generate plan then execute each step.
    pub async fn run(&self, task: &str) -> error::Result<PlanningResult> {
        // Step 1: Generate plan
        let plan_prompt = format!(
            r#"You are a planning agent. Decompose this task into a structured plan.
Output ONLY valid JSON in this format:
{{
  "goal": "the overall goal",
  "steps": [
    {{"id": "step1", "description": "what to do", "depends_on": []}},
    {{"id": "step2", "description": "next step", "depends_on": ["step1"]}}
  ]
}}

Maximum {max} steps. Be specific and actionable.

Task: {task}"#,
            max = self.max_plan_steps,
            task = task
        );

        let plan_output = self.agent.run(vec![Message::user(plan_prompt)]).await?;

        let mut plan = Self::parse_plan(&plan_output.text)?;

        // Step 2: Execute each step in dependency order
        let mut last_output = plan_output;
        while !plan.is_complete() && !plan.has_failures() {
            let ready = plan.ready_steps();
            if ready.is_empty() {
                break;
            }

            // Execute first ready step (sequential for simplicity)
            let step_id = ready[0].id.clone();
            let step_desc = ready[0].description.clone();
            plan.set_status(&step_id, PlanStepStatus::InProgress);

            let context = format!(
                "You are executing step '{}' of a plan.\nGoal: {}\nStep description: {}\n\nPrevious context: {}",
                step_id, plan.goal, step_desc, last_output.text
            );

            match self.agent.run(vec![Message::user(context)]).await {
                Ok(output) => {
                    plan.set_result(&step_id, output.text.clone());
                    plan.set_status(&step_id, PlanStepStatus::Done);
                    last_output = output;
                }
                Err(e) => {
                    plan.set_result(&step_id, format!("Error: {e}"));
                    plan.set_status(&step_id, PlanStepStatus::Failed);
                }
            }
        }

        Ok(PlanningResult {
            plan,
            final_output: last_output,
        })
    }
}
