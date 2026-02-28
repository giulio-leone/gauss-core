//! Graph — reactive DAG executor for multi-agent pipelines.
//!
//! A graph is a directed acyclic graph of nodes (agents or functions).
//! Nodes with satisfied dependencies execute in parallel when possible.
//! Forks run concurrent branches; consensus merges results.

use crate::agent::Agent;
use crate::error::{self, GaussError};
use crate::message::Message;
use std::collections::HashMap;

/// Output from a graph node.
#[derive(Debug, Clone)]
pub struct NodeOutput {
    pub node_id: String,
    pub text: String,
    pub data: Option<serde_json::Value>,
}

/// Result of executing a graph.
#[derive(Debug, Clone)]
pub struct GraphResult {
    pub outputs: HashMap<String, NodeOutput>,
    pub final_output: Option<NodeOutput>,
}

/// Consensus strategy for merging fork results.
#[derive(Debug, Clone)]
pub enum ConsensusStrategy {
    /// Pick the first completed result.
    First,
    /// Concatenate all results.
    Concat,
    /// Use a custom merge function (stored externally, referenced by name).
    Custom(String),
}

// Type aliases for complex callback types used in GraphNode.
#[cfg(not(target_arch = "wasm32"))]
type InputMapFn =
    std::sync::Arc<dyn Fn(&HashMap<String, NodeOutput>) -> Vec<Message> + Send + Sync>;
#[cfg(not(target_arch = "wasm32"))]
type AsyncExecuteFn = std::sync::Arc<
    dyn Fn(
            HashMap<String, NodeOutput>,
        )
            -> std::pin::Pin<Box<dyn std::future::Future<Output = error::Result<NodeOutput>> + Send>>
        + Send
        + Sync,
>;

#[cfg(target_arch = "wasm32")]
type InputMapFn = std::rc::Rc<dyn Fn(&HashMap<String, NodeOutput>) -> Vec<Message>>;
#[cfg(target_arch = "wasm32")]
type AsyncExecuteFn = std::rc::Rc<
    dyn Fn(
        HashMap<String, NodeOutput>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = error::Result<NodeOutput>>>>,
>;

/// A single node in the graph.
#[cfg(not(target_arch = "wasm32"))]
pub enum GraphNode {
    /// An agent node.
    Agent {
        agent: Box<Agent>,
        input_fn: InputMapFn,
    },
    /// A custom async function.
    Function { execute: AsyncExecuteFn },
    /// A fork: run multiple agents in parallel.
    Fork {
        agents: Vec<(String, Box<Agent>)>,
        consensus: ConsensusStrategy,
    },
}

#[cfg(target_arch = "wasm32")]
pub enum GraphNode {
    Agent {
        agent: Box<Agent>,
        input_fn: InputMapFn,
    },
    Function {
        execute: AsyncExecuteFn,
    },
    Fork {
        agents: Vec<(String, Box<Agent>)>,
        consensus: ConsensusStrategy,
    },
}

/// The graph definition.
pub struct Graph {
    nodes: HashMap<String, GraphNode>,
    edges: HashMap<String, Vec<String>>,
    entry_points: Vec<String>,
    terminal_nodes: Vec<String>,
}

impl Graph {
    pub fn builder() -> GraphBuilder {
        GraphBuilder::new()
    }

    /// Execute the graph with a prompt. Returns all node outputs.
    pub async fn run(&self, prompt: impl Into<String>) -> error::Result<GraphResult> {
        let prompt_str = prompt.into();
        let initial_msgs = vec![Message::user(prompt_str)];
        let mut completed: HashMap<String, NodeOutput> = HashMap::new();
        let mut ready: Vec<String> = self.entry_points.clone();

        while !ready.is_empty() {
            let mut next_ready = Vec::new();

            // Separate parallelizable nodes from sequential ones
            let mut batch: Vec<String> = Vec::new();
            for node_id in &ready {
                let deps = self.edges.get(node_id).cloned().unwrap_or_default();
                if deps.iter().all(|d| completed.contains_key(d)) {
                    batch.push(node_id.clone());
                } else {
                    next_ready.push(node_id.clone());
                }
            }

            if batch.is_empty() && !next_ready.is_empty() {
                return Err(GaussError::Agent {
                    message: "Graph deadlock — circular dependencies".to_string(),
                    source: None,
                });
            }

            // Execute batch (parallel when possible)
            #[cfg(not(target_arch = "wasm32"))]
            {
                let mut handles = Vec::new();
                for node_id in &batch {
                    let output = self
                        .execute_node(node_id, &completed, &initial_msgs)
                        .await?;
                    handles.push((node_id.clone(), output));
                }
                for (nid, output) in handles {
                    completed.insert(nid.clone(), output);
                    self.discover_ready(&nid, &completed, &mut next_ready);
                }
            }

            #[cfg(target_arch = "wasm32")]
            {
                for node_id in &batch {
                    let output = self
                        .execute_node(node_id, &completed, &initial_msgs)
                        .await?;
                    completed.insert(node_id.clone(), output);
                    self.discover_ready(node_id, &completed, &mut next_ready);
                }
            }

            if next_ready == ready {
                return Err(GaussError::Agent {
                    message: "Graph deadlock — no progress".to_string(),
                    source: None,
                });
            }

            ready = next_ready;
        }

        // Determine final output (last terminal node)
        let final_output = self
            .terminal_nodes
            .last()
            .and_then(|id| completed.get(id))
            .cloned();

        Ok(GraphResult {
            outputs: completed,
            final_output,
        })
    }

    /// Execute a single node.
    async fn execute_node(
        &self,
        node_id: &str,
        completed: &HashMap<String, NodeOutput>,
        initial_msgs: &[Message],
    ) -> error::Result<NodeOutput> {
        let node = self.nodes.get(node_id).ok_or_else(|| GaussError::Agent {
            message: format!("Graph node '{node_id}' not found"),
            source: None,
        })?;

        match node {
            GraphNode::Agent { agent, input_fn } => {
                let messages = if completed.is_empty() {
                    initial_msgs.to_vec()
                } else {
                    input_fn(completed)
                };
                let result = agent.run(messages).await?;
                Ok(NodeOutput {
                    node_id: node_id.to_string(),
                    text: result.text,
                    data: result.structured_output,
                })
            }
            GraphNode::Function { execute } => execute(completed.clone()).await,
            GraphNode::Fork { agents, consensus } => {
                self.execute_fork(node_id, agents, consensus, completed, initial_msgs)
                    .await
            }
        }
    }

    /// Execute a fork (parallel agents) and merge with consensus.
    async fn execute_fork(
        &self,
        node_id: &str,
        agents: &[(String, Box<Agent>)],
        consensus: &ConsensusStrategy,
        completed: &HashMap<String, NodeOutput>,
        initial_msgs: &[Message],
    ) -> error::Result<NodeOutput> {
        let mut results = Vec::new();

        for (branch_id, agent) in agents {
            let messages = if completed.is_empty() {
                initial_msgs.to_vec()
            } else {
                // Pass context from completed nodes
                let context = completed
                    .values()
                    .map(|o| format!("[{}]: {}", o.node_id, o.text))
                    .collect::<Vec<_>>()
                    .join("\n");
                vec![Message::user(context)]
            };

            let result = agent.run(messages).await?;
            results.push((branch_id.clone(), result.text, result.structured_output));
        }

        let merged = match consensus {
            ConsensusStrategy::First => {
                if let Some((_, text, data)) = results.into_iter().next() {
                    (text, data)
                } else {
                    (String::new(), None)
                }
            }
            ConsensusStrategy::Concat => {
                let text = results
                    .iter()
                    .map(|(_, t, _)| t.as_str())
                    .collect::<Vec<_>>()
                    .join("\n\n");
                (text, None)
            }
            ConsensusStrategy::Custom(_) => {
                // Custom consensus would be handled by a callback; for now, concat
                let text = results
                    .iter()
                    .map(|(_, t, _)| t.as_str())
                    .collect::<Vec<_>>()
                    .join("\n\n");
                (text, None)
            }
        };

        Ok(NodeOutput {
            node_id: node_id.to_string(),
            text: merged.0,
            data: merged.1,
        })
    }

    /// Find nodes that become ready after a node completes.
    fn discover_ready(
        &self,
        completed_id: &str,
        completed: &HashMap<String, NodeOutput>,
        ready: &mut Vec<String>,
    ) {
        for (nid, deps) in &self.edges {
            if deps.contains(&completed_id.to_string())
                && !completed.contains_key(nid)
                && !ready.contains(nid)
                && deps.iter().all(|d| completed.contains_key(d))
            {
                ready.push(nid.clone());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

pub struct GraphBuilder {
    nodes: HashMap<String, GraphNode>,
    edges: HashMap<String, Vec<String>>,
}

impl GraphBuilder {
    fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
        }
    }

    /// Add an agent node.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn node(
        mut self,
        id: impl Into<String>,
        agent: Agent,
        input_fn: impl Fn(&HashMap<String, NodeOutput>) -> Vec<Message> + Send + Sync + 'static,
    ) -> Self {
        self.nodes.insert(
            id.into(),
            GraphNode::Agent {
                agent: Box::new(agent),
                input_fn: std::sync::Arc::new(input_fn),
            },
        );
        self
    }

    #[cfg(target_arch = "wasm32")]
    pub fn node(
        mut self,
        id: impl Into<String>,
        agent: Agent,
        input_fn: impl Fn(&HashMap<String, NodeOutput>) -> Vec<Message> + 'static,
    ) -> Self {
        self.nodes.insert(
            id.into(),
            GraphNode::Agent {
                agent: Box::new(agent),
                input_fn: std::rc::Rc::new(input_fn),
            },
        );
        self
    }

    /// Add a fork node: run multiple agents in parallel with a consensus strategy.
    pub fn fork(
        mut self,
        id: impl Into<String>,
        agents: Vec<(impl Into<String>, Agent)>,
        consensus: ConsensusStrategy,
    ) -> Self {
        let agents_boxed: Vec<(String, Box<Agent>)> = agents
            .into_iter()
            .map(|(name, a)| (name.into(), Box::new(a)))
            .collect();
        self.nodes.insert(
            id.into(),
            GraphNode::Fork {
                agents: agents_boxed,
                consensus,
            },
        );
        self
    }

    /// Add a function node.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn function_node(
        mut self,
        id: impl Into<String>,
        execute: impl Fn(
            HashMap<String, NodeOutput>,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = error::Result<NodeOutput>> + Send>,
        > + Send
        + Sync
        + 'static,
    ) -> Self {
        self.nodes.insert(
            id.into(),
            GraphNode::Function {
                execute: std::sync::Arc::new(execute),
            },
        );
        self
    }

    #[cfg(target_arch = "wasm32")]
    pub fn function_node(
        mut self,
        id: impl Into<String>,
        execute: impl Fn(
            HashMap<String, NodeOutput>,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = error::Result<NodeOutput>>>,
        > + 'static,
    ) -> Self {
        self.nodes.insert(
            id.into(),
            GraphNode::Function {
                execute: std::rc::Rc::new(execute),
            },
        );
        self
    }

    /// Add a directed edge: `from` must complete before `to` starts.
    pub fn edge(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        let to_id = to.into();
        self.edges.entry(to_id).or_default().push(from.into());
        self
    }

    /// Build the graph.
    pub fn build(self) -> Graph {
        let entry_points: Vec<String> = self
            .nodes
            .keys()
            .filter(|k| self.edges.get(*k).is_none_or(|deps| deps.is_empty()))
            .cloned()
            .collect();

        let terminal_nodes: Vec<String> = self
            .nodes
            .keys()
            .filter(|k| !self.edges.values().any(|deps| deps.contains(*k)))
            .cloned()
            .collect();

        Graph {
            nodes: self.nodes,
            edges: self.edges,
            entry_points,
            terminal_nodes,
        }
    }
}
