//! Agent Network — multi-agent communication, delegation, and orchestration.
//!
//! Extends the basic Team model with graph-based agent networks,
//! message routing, sub-agent delegation, and supervisor/worker patterns.

use crate::agent::Agent;
use crate::error;
use crate::message::Message;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Agent Card (A2A protocol)
// ---------------------------------------------------------------------------

/// Agent capability descriptor (following Google A2A spec).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCard {
    pub name: String,
    pub description: String,
    /// List of capabilities/skills this agent provides.
    pub capabilities: Vec<String>,
    /// URL endpoint (for remote agents).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Supported input/output types.
    #[serde(default)]
    pub input_modes: Vec<String>,
    #[serde(default)]
    pub output_modes: Vec<String>,
}

// ---------------------------------------------------------------------------
// Network Message
// ---------------------------------------------------------------------------

/// A message routed between agents in a network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMessage {
    pub from: String,
    pub to: String,
    pub content: String,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Result of a delegation to a sub-agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelegationResult {
    pub agent_name: String,
    pub result_text: String,
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

// ---------------------------------------------------------------------------
// Agent Network Node
// ---------------------------------------------------------------------------

/// A node in the agent network — wraps an Agent with routing metadata.
pub struct AgentNode {
    pub agent: Agent,
    pub card: AgentCard,
    /// Names of agents this node can delegate to.
    pub connections: Vec<String>,
}

// ---------------------------------------------------------------------------
// Agent Network
// ---------------------------------------------------------------------------

/// A graph of connected agents with routing and delegation capabilities.
pub struct AgentNetwork {
    nodes: HashMap<String, AgentNode>,
    /// Optional supervisor agent name.
    supervisor: Option<String>,
}

impl Default for AgentNetwork {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentNetwork {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            supervisor: None,
        }
    }

    /// Add an agent to the network.
    pub fn add_agent(&mut self, node: AgentNode) {
        self.nodes.insert(node.card.name.clone(), node);
    }

    /// Remove an agent from the network.
    pub fn remove_agent(&mut self, name: &str) -> Option<AgentNode> {
        let node = self.nodes.remove(name);
        // Clean up connections pointing to removed agent
        for n in self.nodes.values_mut() {
            n.connections.retain(|c| c != name);
        }
        node
    }

    /// Set the supervisor agent.
    pub fn set_supervisor(&mut self, name: impl Into<String>) {
        self.supervisor = Some(name.into());
    }

    /// Get all agent cards in the network.
    pub fn agent_cards(&self) -> Vec<&AgentCard> {
        self.nodes.values().map(|n| &n.card).collect()
    }

    /// Find the best agent for a task based on capability matching.
    pub fn route(&self, capabilities: &[String]) -> Option<&str> {
        let mut best_match: Option<(&str, usize)> = None;

        for (name, node) in &self.nodes {
            let matches = capabilities
                .iter()
                .filter(|c| node.card.capabilities.contains(c))
                .count();
            if matches > 0 && (best_match.is_none() || matches > best_match.unwrap().1) {
                best_match = Some((name.as_str(), matches));
            }
        }

        best_match.map(|(name, _)| name)
    }

    /// Delegate a task to a specific agent in the network.
    pub async fn delegate(
        &self,
        agent_name: &str,
        messages: Vec<Message>,
    ) -> error::Result<DelegationResult> {
        let node = self.nodes.get(agent_name).ok_or_else(|| {
            error::GaussError::internal(format!("Agent '{}' not found in network", agent_name))
        })?;

        match node.agent.run(messages).await {
            Ok(output) => Ok(DelegationResult {
                agent_name: agent_name.to_string(),
                result_text: output.text,
                success: true,
                error: None,
            }),
            Err(e) => Ok(DelegationResult {
                agent_name: agent_name.to_string(),
                result_text: String::new(),
                success: false,
                error: Some(e.to_string()),
            }),
        }
    }

    /// Run the supervisor pattern: supervisor agent decides which sub-agent to delegate to.
    pub async fn run_supervised(&self, messages: Vec<Message>) -> error::Result<DelegationResult> {
        let supervisor_name = self.supervisor.as_deref().ok_or_else(|| {
            error::GaussError::internal("No supervisor agent configured".to_string())
        })?;

        // First, run the supervisor to decide which agent to use.
        let supervisor_node = self.nodes.get(supervisor_name).ok_or_else(|| {
            error::GaussError::internal(format!(
                "Supervisor '{}' not found in network",
                supervisor_name
            ))
        })?;

        let cards: Vec<String> = self
            .nodes
            .iter()
            .filter(|(name, _)| name.as_str() != supervisor_name)
            .map(|(_, node)| {
                format!(
                    "- {}: {} (capabilities: {})",
                    node.card.name,
                    node.card.description,
                    node.card.capabilities.join(", ")
                )
            })
            .collect();

        let routing_prompt = format!(
            "You are a supervisor agent. Available sub-agents:\n{}\n\n\
             Based on the user's request, respond with ONLY the name of the best sub-agent to handle this task.",
            cards.join("\n")
        );

        let mut routing_messages = vec![Message::system(routing_prompt)];
        routing_messages.extend(messages.clone());

        let supervisor_output = supervisor_node.agent.run(routing_messages).await?;
        let chosen_agent = supervisor_output.text.trim().to_string();

        // Delegate to the chosen agent
        if self.nodes.contains_key(&chosen_agent) {
            self.delegate(&chosen_agent, messages).await
        } else {
            // Fallback: try to find partial match
            let found = self
                .nodes
                .keys()
                .find(|k| chosen_agent.to_lowercase().contains(&k.to_lowercase()));

            match found {
                Some(name) => self.delegate(name, messages).await,
                None => Err(error::GaussError::internal(format!(
                    "Supervisor chose unknown agent: '{}'",
                    chosen_agent
                ))),
            }
        }
    }

    /// Run all agents in parallel and aggregate results.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn broadcast(&self, messages: Vec<Message>) -> error::Result<Vec<DelegationResult>> {
        let mut handles = Vec::new();

        for name in self.nodes.keys() {
            let name = name.clone();
            let msgs = messages.clone();
            // We can't easily spawn here without Shared agents,
            // so we run sequentially as a safe default.
            handles.push((name, msgs));
        }

        let mut results = Vec::new();
        for (name, msgs) in handles {
            results.push(self.delegate(&name, msgs).await?);
        }
        Ok(results)
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn broadcast(&self, messages: Vec<Message>) -> error::Result<Vec<DelegationResult>> {
        let mut results = Vec::new();
        for name in self.nodes.keys() {
            results.push(self.delegate(name, messages.clone()).await?);
        }
        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// Network Builder
// ---------------------------------------------------------------------------

/// Builder for constructing an AgentNetwork.
pub struct AgentNetworkBuilder {
    network: AgentNetwork,
}

impl Default for AgentNetworkBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentNetworkBuilder {
    pub fn new() -> Self {
        Self {
            network: AgentNetwork::new(),
        }
    }

    pub fn agent(mut self, node: AgentNode) -> Self {
        self.network.add_agent(node);
        self
    }

    pub fn supervisor(mut self, name: impl Into<String>) -> Self {
        self.network.set_supervisor(name);
        self
    }

    pub fn build(self) -> AgentNetwork {
        self.network
    }
}
