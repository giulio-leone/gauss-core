use crate::provider::{get_provider, PROVIDERS};
use crate::registry::HandleRegistry;
use crate::types::ToolDef;
use gauss_core::agent::Agent as RustAgent;
use gauss_core::message::Message as RustMessage;
use gauss_core::tool::Tool as RustTool;
use gauss_core::{team};
use napi::bindgen_prelude::*;
use serde_json::json;

// ============ Graph ============

struct GraphState {
    nodes: Vec<GraphNodeDefKind>,
    edges: Vec<(String, String)>,
}

enum GraphNodeDefKind {
    Agent(GraphNodeDef),
    Fork(GraphForkDef),
}

struct GraphNodeDef {
    id: String,
    agent_name: String,
    provider_handle: u32,
    instructions: Option<String>,
    tools: Vec<(String, String, Option<serde_json::Value>)>,
}

struct GraphForkDef {
    id: String,
    agents: Vec<GraphNodeDef>,
    consensus: String,
}

static GRAPHS: HandleRegistry<GraphState> = HandleRegistry::new();

#[napi]
pub fn create_graph() -> u32 {
    GRAPHS.insert(GraphState {
        nodes: vec![],
        edges: vec![],
    })
}

#[napi]
pub fn graph_add_node(
    handle: u32,
    node_id: String,
    agent_name: String,
    provider_handle: u32,
    instructions: Option<String>,
    tools: Vec<ToolDef>,
) -> Result<()> {
    GRAPHS.with_mut(handle, |state| {
        state.nodes.push(GraphNodeDefKind::Agent(GraphNodeDef {
            id: node_id,
            agent_name,
            provider_handle,
            instructions,
            tools: tools
                .into_iter()
                .map(|t| (t.name, t.description, t.parameters))
                .collect(),
        }));
        Ok(())
    })
}

#[napi]
pub fn graph_add_edge(handle: u32, from: String, to: String) -> Result<()> {
    GRAPHS.with_mut(handle, |state| {
        state.edges.push((from, to));
        Ok(())
    })
}

#[napi(object)]
pub struct ForkAgentDef {
    pub agent_name: String,
    pub provider_handle: u32,
    pub instructions: Option<String>,
}

#[napi]
pub fn graph_add_fork_node(
    handle: u32,
    node_id: String,
    agents: Vec<ForkAgentDef>,
    consensus: String,
) -> Result<()> {
    GRAPHS.with_mut(handle, |state| {
        state.nodes.push(GraphNodeDefKind::Fork(GraphForkDef {
            id: node_id,
            agents: agents
                .into_iter()
                .map(|a| GraphNodeDef {
                    id: a.agent_name.clone(),
                    agent_name: a.agent_name,
                    provider_handle: a.provider_handle,
                    instructions: a.instructions,
                    tools: vec![],
                })
                .collect(),
            consensus,
        }));
        Ok(())
    })
}

#[napi]
pub async fn graph_run(handle: u32, prompt: String) -> Result<serde_json::Value> {
    enum NodeSnapshot {
        Agent {
            id: String,
            agent_name: String,
            provider_handle: u32,
            instructions: Option<String>,
            tools: Vec<(String, String, Option<serde_json::Value>)>,
        },
        Fork {
            id: String,
            agents: Vec<(String, u32, Option<String>)>,
            consensus: String,
        },
    }

    let (snapshots, edges) = GRAPHS.get(handle, |s| {
        let snaps: Vec<NodeSnapshot> = s
            .nodes
            .iter()
            .map(|n| match n {
                GraphNodeDefKind::Agent(a) => NodeSnapshot::Agent {
                    id: a.id.clone(),
                    agent_name: a.agent_name.clone(),
                    provider_handle: a.provider_handle,
                    instructions: a.instructions.clone(),
                    tools: a.tools.clone(),
                },
                GraphNodeDefKind::Fork(f) => NodeSnapshot::Fork {
                    id: f.id.clone(),
                    agents: f
                        .agents
                        .iter()
                        .map(|a| (a.agent_name.clone(), a.provider_handle, a.instructions.clone()))
                        .collect(),
                    consensus: f.consensus.clone(),
                },
            })
            .collect();
        (snaps, s.edges.clone())
    })?;

    let mut builder = gauss_core::Graph::builder();

    for snap in &snapshots {
        match snap {
            NodeSnapshot::Agent {
                id,
                agent_name,
                provider_handle,
                instructions,
                tools,
            } => {
                let provider = {
                    let provs = PROVIDERS.raw().lock().expect("registry mutex poisoned");
                    provs.get(provider_handle).cloned().ok_or_else(|| {
                        napi::Error::from_reason(format!("Provider {provider_handle} not found"))
                    })?
                };
                let mut ab = RustAgent::builder(agent_name.clone(), provider);
                if let Some(instr) = instructions {
                    ab = ab.instructions(instr.clone());
                }
                for (name, desc, params) in tools {
                    let mut tb = RustTool::builder(name, desc);
                    if let Some(p) = params {
                        tb = tb.parameters_json(p.clone());
                    }
                    ab = ab.tool(tb.build());
                }
                let agent = ab.build();
                let nid = id.clone();
                builder = builder.node(nid, agent, |_deps| {
                    vec![RustMessage::user("Continue based on prior context.")]
                });
            }
            NodeSnapshot::Fork {
                id,
                agents,
                consensus,
            } => {
                let mut fork_agents: Vec<(String, gauss_core::Agent)> = Vec::new();
                for (name, prov_handle, instructions) in agents {
                    let provider = {
                        let provs = PROVIDERS.raw().lock().expect("registry mutex poisoned");
                        provs.get(prov_handle).cloned().ok_or_else(|| {
                            napi::Error::from_reason(format!("Provider {prov_handle} not found"))
                        })?
                    };
                    let mut ab = RustAgent::builder(name.clone(), provider);
                    if let Some(instr) = instructions {
                        ab = ab.instructions(instr.clone());
                    }
                    fork_agents.push((name.clone(), ab.build()));
                }
                let strategy = match consensus.as_str() {
                    "first" => gauss_core::graph::ConsensusStrategy::First,
                    _ => gauss_core::graph::ConsensusStrategy::Concat,
                };
                builder = builder.fork(id.clone(), fork_agents, strategy);
            }
        }
    }

    for (from, to) in &edges {
        builder = builder.edge(from.clone(), to.clone());
    }

    let graph = builder.build();
    let result = graph
        .run(prompt)
        .await
        .map_err(|e| napi::Error::from_reason(format!("{e}")))?;

    let outputs: serde_json::Value = result
        .outputs
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                json!({ "node_id": v.node_id, "text": v.text, "data": v.data }),
            )
        })
        .collect::<serde_json::Map<String, serde_json::Value>>()
        .into();

    Ok(json!({
        "outputs": outputs,
        "final_text": result.final_output.map(|o| o.text),
    }))
}

#[napi]
pub fn destroy_graph(handle: u32) -> Result<()> {
    GRAPHS.remove(handle)?;
    Ok(())
}

// ============ Workflow ============

struct WorkflowState {
    steps: Vec<WorkflowStepDef>,
    dependencies: Vec<(String, String)>,
}

struct WorkflowStepDef {
    id: String,
    agent_name: String,
    provider_handle: u32,
    instructions: Option<String>,
    tools: Vec<(String, String, Option<serde_json::Value>)>,
}

static WORKFLOWS: HandleRegistry<WorkflowState> = HandleRegistry::new();

#[napi]
pub fn create_workflow() -> u32 {
    WORKFLOWS.insert(WorkflowState {
        steps: vec![],
        dependencies: vec![],
    })
}

#[napi]
pub fn workflow_add_step(
    handle: u32,
    step_id: String,
    agent_name: String,
    provider_handle: u32,
    instructions: Option<String>,
    tools: Vec<ToolDef>,
) -> Result<()> {
    WORKFLOWS.with_mut(handle, |state| {
        state.steps.push(WorkflowStepDef {
            id: step_id,
            agent_name,
            provider_handle,
            instructions,
            tools: tools
                .into_iter()
                .map(|t| (t.name, t.description, t.parameters))
                .collect(),
        });
        Ok(())
    })
}

#[napi]
pub fn workflow_add_dependency(handle: u32, step_id: String, depends_on: String) -> Result<()> {
    WORKFLOWS.with_mut(handle, |state| {
        state.dependencies.push((step_id, depends_on));
        Ok(())
    })
}

#[napi]
pub async fn workflow_run(handle: u32, prompt: String) -> Result<serde_json::Value> {
    let (step_defs, deps) = WORKFLOWS.get(handle, |s| {
        let steps: Vec<_> = s
            .steps
            .iter()
            .map(|st| {
                (
                    st.id.clone(),
                    st.agent_name.clone(),
                    st.provider_handle,
                    st.instructions.clone(),
                    st.tools.clone(),
                )
            })
            .collect();
        (steps, s.dependencies.clone())
    })?;

    let mut builder = gauss_core::Workflow::builder();

    for (step_id, agent_name, prov_handle, instructions, tools) in &step_defs {
        let provider = {
            let provs = PROVIDERS.raw().lock().expect("registry mutex poisoned");
            provs.get(prov_handle).cloned().ok_or_else(|| {
                napi::Error::from_reason(format!("Provider {prov_handle} not found"))
            })?
        };

        let mut agent_builder = RustAgent::builder(agent_name.clone(), provider);
        if let Some(instr) = instructions {
            agent_builder = agent_builder.instructions(instr.clone());
        }
        for (name, desc, params) in tools {
            let mut tb = RustTool::builder(name, desc);
            if let Some(p) = params {
                tb = tb.parameters_json(p.clone());
            }
            agent_builder = agent_builder.tool(tb.build());
        }
        let agent = agent_builder.build();

        builder = builder.agent_step(step_id.clone(), agent, |_completed| {
            vec![RustMessage::user("Continue.")]
        });
    }

    for (step_id, depends_on) in &deps {
        builder = builder.dependency(step_id.clone(), depends_on.clone());
    }

    let workflow = builder.build();
    let initial = vec![RustMessage::user(prompt)];
    let result = workflow
        .run(initial)
        .await
        .map_err(|e| napi::Error::from_reason(format!("{e}")))?;

    let outputs: serde_json::Value = result
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                json!({ "step_id": v.step_id, "text": v.text, "data": v.data }),
            )
        })
        .collect::<serde_json::Map<String, serde_json::Value>>()
        .into();

    Ok(json!({ "steps": outputs }))
}

#[napi]
pub fn destroy_workflow(handle: u32) -> Result<()> {
    WORKFLOWS.remove(handle)?;
    Ok(())
}

// ============ Team ============

#[derive(Clone)]
struct TeamAgentDef {
    name: String,
    provider_handle: u32,
    instructions: Option<String>,
}

#[derive(Clone)]
struct TeamState {
    name: String,
    strategy: String,
    agents: Vec<TeamAgentDef>,
}

static TEAMS: HandleRegistry<TeamState> = HandleRegistry::new();

#[napi]
pub fn create_team(name: String) -> u32 {
    TEAMS.insert(TeamState {
        name,
        strategy: "sequential".to_string(),
        agents: Vec::new(),
    })
}

#[napi]
pub fn team_add_agent(
    handle: u32,
    agent_name: String,
    provider_handle: u32,
    instructions: Option<String>,
) -> Result<()> {
    TEAMS.with_mut(handle, |state| {
        state.agents.push(TeamAgentDef {
            name: agent_name,
            provider_handle,
            instructions,
        });
        Ok(())
    })
}

#[napi]
pub fn team_set_strategy(handle: u32, strategy: String) -> Result<()> {
    TEAMS.with_mut(handle, |state| {
        match strategy.as_str() {
            "sequential" | "parallel" => {
                state.strategy = strategy;
                Ok(())
            }
            _ => Err(napi::Error::from_reason(format!(
                "Unknown strategy: {}. Use 'sequential' or 'parallel'",
                strategy
            ))),
        }
    })
}

#[napi]
pub async fn team_run(handle: u32, messages_json: String) -> Result<serde_json::Value> {
    let state = TEAMS.get(handle, |s| s.clone())?;

    let strategy = match state.strategy.as_str() {
        "parallel" => team::Strategy::Parallel,
        _ => team::Strategy::Sequential,
    };

    let mut builder = team::Team::builder(&state.name).strategy(strategy);

    for agent_def in &state.agents {
        let provider = get_provider(agent_def.provider_handle)?;
        let mut agent_builder = gauss_core::agent::Agent::builder(&agent_def.name, provider);
        if let Some(ref instr) = agent_def.instructions {
            agent_builder = agent_builder.instructions(instr.clone());
        }
        builder = builder.agent(agent_builder.build());
    }

    let team_instance = builder.build();

    let messages: Vec<RustMessage> = serde_json::from_str(&messages_json)
        .map_err(|e| napi::Error::from_reason(format!("Invalid messages JSON: {e}")))?;

    let output = team_instance
        .run(messages)
        .await
        .map_err(|e| napi::Error::from_reason(format!("Team run error: {e}")))?;

    let results: Vec<serde_json::Value> = output
        .results
        .iter()
        .map(|r| {
            json!({
                "text": r.text,
                "steps": r.steps,
                "inputTokens": r.usage.input_tokens,
                "outputTokens": r.usage.output_tokens,
            })
        })
        .collect();

    Ok(json!({
        "finalText": output.final_text,
        "results": results,
    }))
}

#[napi]
pub fn destroy_team(handle: u32) -> Result<()> {
    TEAMS.remove(handle)?;
    Ok(())
}
