//! Team — multi-agent coordination patterns.
//!
//! Orchestrates multiple agents with different strategies:
//! - Sequential: agents run in order, each seeing previous outputs
//! - Parallel: agents run concurrently, results merged
//! - Supervisor: a lead agent delegates to specialized agents

use crate::agent::{Agent, AgentOutput};
use crate::error::{self, GaussError};
use crate::message::Message;

/// Team coordination strategy.
#[derive(Debug, Clone)]
pub enum Strategy {
    /// Agents run in sequence. Each sees the previous agent's output.
    Sequential,
    /// Agents run in parallel. Results are collected.
    Parallel,
}

/// Result from a team execution.
#[derive(Debug)]
pub struct TeamOutput {
    pub results: Vec<AgentOutput>,
    pub final_text: String,
}

/// A team of agents.
pub struct Team {
    name: String,
    agents: Vec<Agent>,
    strategy: Strategy,
}

impl Team {
    pub fn builder(name: impl Into<String>) -> TeamBuilder {
        TeamBuilder {
            name: name.into(),
            agents: Vec::new(),
            strategy: Strategy::Sequential,
        }
    }

    /// Run the team with initial messages.
    pub async fn run(&self, messages: Vec<Message>) -> error::Result<TeamOutput> {
        match self.strategy {
            Strategy::Sequential => self.run_sequential(messages).await,
            Strategy::Parallel => self.run_parallel(messages).await,
        }
    }

    async fn run_sequential(&self, messages: Vec<Message>) -> error::Result<TeamOutput> {
        if self.agents.is_empty() {
            return Err(GaussError::Agent {
                message: format!("Team '{}' has no agents", self.name),
                source: None,
            });
        }

        let mut results = Vec::new();
        let mut current_messages = messages;

        for agent in &self.agents {
            let output = agent.run(current_messages).await?;
            current_messages = vec![Message::user(&output.text)];
            results.push(output);
        }

        let final_text = results.last().map(|r| r.text.clone()).unwrap_or_default();

        Ok(TeamOutput {
            results,
            final_text,
        })
    }

    #[cfg(feature = "native")]
    async fn run_parallel(&self, messages: Vec<Message>) -> error::Result<TeamOutput> {
        if self.agents.is_empty() {
            return Err(GaussError::Agent {
                message: format!("Team '{}' has no agents", self.name),
                source: None,
            });
        }

        let mut handles = Vec::new();

        for agent in &self.agents {
            let msgs = messages.clone();
            let agent_clone = agent.clone();
            handles.push(tokio::spawn(async move { agent_clone.run(msgs).await }));
        }

        let mut results = Vec::new();
        for handle in handles {
            let output = handle
                .await
                .map_err(|e| GaussError::Agent {
                    message: format!("Team task failed: {e}"),
                    source: None,
                })?
                .map_err(|e| GaussError::Agent {
                    message: format!("Agent failed: {e}"),
                    source: None,
                })?;
            results.push(output);
        }

        let final_text = results
            .iter()
            .map(|r| r.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n---\n\n");

        Ok(TeamOutput {
            results,
            final_text,
        })
    }

    #[cfg(not(feature = "native"))]
    async fn run_parallel(&self, messages: Vec<Message>) -> error::Result<TeamOutput> {
        // Without tokio, fall back to sequential execution
        self.run_sequential(messages).await
    }
}

/// Builder for Team.
pub struct TeamBuilder {
    name: String,
    agents: Vec<Agent>,
    strategy: Strategy,
}

impl TeamBuilder {
    /// Add an agent to the team.
    pub fn agent(mut self, agent: Agent) -> Self {
        self.agents.push(agent);
        self
    }

    /// Set the coordination strategy.
    pub fn strategy(mut self, strategy: Strategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Build the team.
    pub fn build(self) -> Team {
        Team {
            name: self.name,
            agents: self.agents,
            strategy: self.strategy,
        }
    }
}
