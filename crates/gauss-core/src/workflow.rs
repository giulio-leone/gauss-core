//! Workflow — composable multi-step agent pipelines.
//!
//! A workflow chains steps (agents, functions, routers) into a DAG.
//! Steps execute when all their dependencies are satisfied.

use crate::agent::Agent;
use crate::error::{self, GaussError};
use crate::message::Message;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

/// Result of a workflow step.
#[derive(Debug, Clone)]
pub struct StepOutput {
    pub step_id: String,
    pub text: String,
    pub data: Option<serde_json::Value>,
}

/// Type alias for step input functions.
pub type StepInputFn = Arc<dyn Fn(&HashMap<String, StepOutput>) -> Vec<Message> + Send + Sync>;

/// Type alias for async step execution functions.
pub type StepExecuteFn = Arc<
    dyn Fn(
            HashMap<String, StepOutput>,
        ) -> Pin<Box<dyn std::future::Future<Output = error::Result<StepOutput>> + Send>>
        + Send
        + Sync,
>;

/// Type alias for router functions.
pub type StepRouteFn = Arc<dyn Fn(&HashMap<String, StepOutput>) -> String + Send + Sync>;

/// A single step in a workflow.
pub enum Step {
    /// Run an agent.
    Agent { agent: Box<Agent>, input_fn: StepInputFn },
    /// Run a custom function.
    Function { execute: StepExecuteFn },
    /// Router — pick a branch based on conditions.
    Router { route_fn: StepRouteFn },
}

/// Workflow definition.
pub struct Workflow {
    steps: HashMap<String, Step>,
    dependencies: HashMap<String, Vec<String>>,
    entry_points: Vec<String>,
}

impl Workflow {
    pub fn builder() -> WorkflowBuilder {
        WorkflowBuilder::new()
    }

    /// Execute the workflow. Returns all step outputs.
    pub async fn run(
        &self,
        initial_messages: Vec<Message>,
    ) -> error::Result<HashMap<String, StepOutput>> {
        let mut completed: HashMap<String, StepOutput> = HashMap::new();
        let mut pending: Vec<String> = self.entry_points.clone();

        // Simple topological execution (serial for now, parallel later)
        while !pending.is_empty() {
            let mut next_pending = Vec::new();

            for step_id in &pending {
                let deps = self.dependencies.get(step_id).cloned().unwrap_or_default();
                let all_deps_done = deps.iter().all(|d| completed.contains_key(d));

                if !all_deps_done {
                    next_pending.push(step_id.clone());
                    continue;
                }

                let step = self.steps.get(step_id).ok_or_else(|| GaussError::Agent {
                    message: format!("Step '{step_id}' not found"),
                    source: None,
                })?;

                let output = match step {
                    Step::Agent { agent, input_fn } => {
                        let messages = if completed.is_empty() {
                            initial_messages.clone()
                        } else {
                            input_fn(&completed)
                        };
                        let result = agent.run(messages).await?;
                        StepOutput {
                            step_id: step_id.clone(),
                            text: result.text,
                            data: result.structured_output,
                        }
                    }
                    Step::Function { execute } => execute(completed.clone()).await?,
                    Step::Router { route_fn } => {
                        let next_step = route_fn(&completed);
                        next_pending.push(next_step.clone());
                        StepOutput {
                            step_id: step_id.clone(),
                            text: next_step,
                            data: None,
                        }
                    }
                };

                completed.insert(step_id.clone(), output);

                // Find steps that depend on this one
                for (sid, deps) in &self.dependencies {
                    if deps.contains(step_id)
                        && !completed.contains_key(sid)
                        && !next_pending.contains(sid)
                    {
                        next_pending.push(sid.clone());
                    }
                }
            }

            if next_pending == pending {
                return Err(GaussError::Agent {
                    message: "Workflow deadlock — circular dependencies".to_string(),
                    source: None,
                });
            }

            pending = next_pending;
        }

        Ok(completed)
    }
}

/// Builder for workflows.
pub struct WorkflowBuilder {
    steps: HashMap<String, Step>,
    dependencies: HashMap<String, Vec<String>>,
}

impl WorkflowBuilder {
    fn new() -> Self {
        Self {
            steps: HashMap::new(),
            dependencies: HashMap::new(),
        }
    }

    /// Add an agent step.
    pub fn agent_step(
        mut self,
        id: impl Into<String>,
        agent: Agent,
        input_fn: impl Fn(&HashMap<String, StepOutput>) -> Vec<Message> + Send + Sync + 'static,
    ) -> Self {
        self.steps.insert(
            id.into(),
            Step::Agent {
                agent: Box::new(agent),
                input_fn: Arc::new(input_fn),
            },
        );
        self
    }

    /// Add a function step.
    pub fn function_step(
        mut self,
        id: impl Into<String>,
        execute: impl Fn(
            HashMap<String, StepOutput>,
        )
            -> Pin<Box<dyn std::future::Future<Output = error::Result<StepOutput>> + Send>>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        self.steps.insert(
            id.into(),
            Step::Function {
                execute: Arc::new(execute),
            },
        );
        self
    }

    /// Add a dependency: `step_id` depends on `depends_on`.
    pub fn dependency(mut self, step_id: impl Into<String>, depends_on: impl Into<String>) -> Self {
        self.dependencies
            .entry(step_id.into())
            .or_default()
            .push(depends_on.into());
        self
    }

    /// Build the workflow.
    pub fn build(self) -> Workflow {
        // Entry points = steps with no dependencies
        let entry_points: Vec<String> = self
            .steps
            .keys()
            .filter(|k| self.dependencies.get(*k).is_none_or(|deps| deps.is_empty()))
            .cloned()
            .collect();

        Workflow {
            steps: self.steps,
            dependencies: self.dependencies,
            entry_points,
        }
    }
}
