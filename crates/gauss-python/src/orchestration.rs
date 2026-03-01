use crate::provider::get_provider;
use crate::registry::{py_err, HandleRegistry};
use gauss_core::agent::Agent as RustAgent;
use gauss_core::message::Message as RustMessage;
use gauss_core::team;
use pyo3::prelude::*;
use serde_json::json;
use std::sync::atomic::{AtomicU32, Ordering};

// ============ Graph ============

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

enum GraphNodeDefKind {
    Agent(GraphNodeDef),
    Fork(GraphForkDef),
}

struct GraphState {
    nodes: Vec<GraphNodeDefKind>,
    edges: Vec<(String, String)>,
}

static GRAPHS: HandleRegistry<GraphState> = HandleRegistry::new();

#[pyfunction]
pub fn create_graph() -> u32 {
    GRAPHS.insert(GraphState {
        nodes: vec![],
        edges: vec![],
    })
}

#[pyfunction]
#[pyo3(signature = (handle, node_id, agent_name, provider_handle, instructions=None, tools_json=None))]
pub fn graph_add_node(
    handle: u32,
    node_id: String,
    agent_name: String,
    provider_handle: u32,
    instructions: Option<String>,
    tools_json: Option<String>,
) -> PyResult<()> {
    let tools: Vec<(String, String, Option<serde_json::Value>)> = match tools_json {
        Some(j) => {
            let parsed: Vec<serde_json::Value> = serde_json::from_str(&j).map_err(py_err)?;
            parsed
                .into_iter()
                .map(|t| {
                    let name = t["name"].as_str().unwrap_or("").to_string();
                    let desc = t["description"].as_str().unwrap_or("").to_string();
                    let params = t.get("parameters").cloned();
                    (name, desc, params)
                })
                .collect()
        }
        None => vec![],
    };
    GRAPHS.with_mut(handle, |state| {
        state.nodes.push(GraphNodeDefKind::Agent(GraphNodeDef {
            id: node_id,
            agent_name,
            provider_handle,
            instructions,
            tools,
        }));
        Ok(())
    })
}

#[pyfunction]
pub fn graph_add_edge(handle: u32, from: String, to: String) -> PyResult<()> {
    GRAPHS.with_mut(handle, |state| {
        state.edges.push((from, to));
        Ok(())
    })
}

#[pyfunction]
#[pyo3(signature = (handle, node_id, agents_json, consensus))]
pub fn graph_add_fork_node(
    handle: u32,
    node_id: String,
    agents_json: String,
    consensus: String,
) -> PyResult<()> {
    let parsed: Vec<serde_json::Value> = serde_json::from_str(&agents_json).map_err(py_err)?;
    let agents = parsed
        .into_iter()
        .map(|a| {
            let name = a["agent_name"].as_str().unwrap_or("").to_string();
            GraphNodeDef {
                id: name.clone(),
                agent_name: name,
                provider_handle: a["provider_handle"].as_u64().unwrap_or(0) as u32,
                instructions: a["instructions"].as_str().map(|s| s.to_string()),
                tools: vec![],
            }
        })
        .collect();
    GRAPHS.with_mut(handle, |state| {
        state.nodes.push(GraphNodeDefKind::Fork(GraphForkDef {
            id: node_id,
            agents,
            consensus,
        }));
        Ok(())
    })
}

#[pyfunction]
pub fn graph_run<'py>(
    py: Python<'py>,
    handle: u32,
    prompt: String,
) -> PyResult<Bound<'py, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
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
                            .map(|a| {
                                (
                                    a.agent_name.clone(),
                                    a.provider_handle,
                                    a.instructions.clone(),
                                )
                            })
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
                    let provider = get_provider(*provider_handle)?;
                    let mut ab = RustAgent::builder(agent_name.clone(), provider);
                    if let Some(instr) = instructions {
                        ab = ab.instructions(instr.clone());
                    }
                    for (name, desc, params) in tools {
                        let mut tb = gauss_core::tool::Tool::builder(name, desc);
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
                        let provider = get_provider(*prov_handle)?;
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
        let result = graph.run(prompt).await.map_err(py_err)?;

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

        let result_json = serde_json::to_string(&json!({
            "outputs": outputs,
            "final_text": result.final_output.map(|o| o.text),
        }))
        .map_err(py_err)?;

        Ok(result_json)
    })
}

#[pyfunction]
pub fn destroy_graph(handle: u32) -> PyResult<()> {
    GRAPHS.remove(handle)
}

// ============ Workflow ============

struct WorkflowStepDef {
    id: String,
    agent_name: String,
    provider_handle: u32,
    instructions: Option<String>,
    tools: Vec<(String, String, Option<serde_json::Value>)>,
}

struct WorkflowState {
    steps: Vec<WorkflowStepDef>,
    dependencies: Vec<(String, String)>,
}

static WORKFLOWS: HandleRegistry<WorkflowState> = HandleRegistry::new();

#[pyfunction]
pub fn create_workflow() -> u32 {
    WORKFLOWS.insert(WorkflowState {
        steps: vec![],
        dependencies: vec![],
    })
}

#[pyfunction]
#[pyo3(signature = (handle, step_id, agent_name, provider_handle, instructions=None, tools_json=None))]
pub fn workflow_add_step(
    handle: u32,
    step_id: String,
    agent_name: String,
    provider_handle: u32,
    instructions: Option<String>,
    tools_json: Option<String>,
) -> PyResult<()> {
    let tools: Vec<(String, String, Option<serde_json::Value>)> = match tools_json {
        Some(j) => {
            let parsed: Vec<serde_json::Value> = serde_json::from_str(&j).map_err(py_err)?;
            parsed
                .into_iter()
                .map(|t| {
                    let name = t["name"].as_str().unwrap_or("").to_string();
                    let desc = t["description"].as_str().unwrap_or("").to_string();
                    let params = t.get("parameters").cloned();
                    (name, desc, params)
                })
                .collect()
        }
        None => vec![],
    };
    WORKFLOWS.with_mut(handle, |state| {
        state.steps.push(WorkflowStepDef {
            id: step_id,
            agent_name,
            provider_handle,
            instructions,
            tools,
        });
        Ok(())
    })
}

#[pyfunction]
pub fn workflow_add_dependency(handle: u32, step_id: String, depends_on: String) -> PyResult<()> {
    WORKFLOWS.with_mut(handle, |state| {
        state.dependencies.push((step_id, depends_on));
        Ok(())
    })
}

#[pyfunction]
pub fn workflow_run<'py>(
    py: Python<'py>,
    handle: u32,
    prompt: String,
) -> PyResult<Bound<'py, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
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
            let provider = get_provider(*prov_handle)?;
            let mut agent_builder = RustAgent::builder(agent_name.clone(), provider);
            if let Some(instr) = instructions {
                agent_builder = agent_builder.instructions(instr.clone());
            }
            for (name, desc, params) in tools {
                let mut tb = gauss_core::tool::Tool::builder(name, desc);
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
        let result = workflow.run(initial).await.map_err(py_err)?;

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

        let result_json = serde_json::to_string(&json!({ "steps": outputs })).map_err(py_err)?;
        Ok(result_json)
    })
}

#[pyfunction]
pub fn destroy_workflow(handle: u32) -> PyResult<()> {
    WORKFLOWS.remove(handle)
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

#[pyfunction]
pub fn create_team(name: String) -> u32 {
    static COUNTER: AtomicU32 = AtomicU32::new(1);
    let handle = COUNTER.fetch_add(1, Ordering::Relaxed);
    TEAMS
        .raw()
        .lock()
        .unwrap()
        .insert(
            handle,
            TeamState {
                name,
                strategy: "sequential".to_string(),
                agents: Vec::new(),
            },
        );
    handle
}

#[pyfunction]
#[pyo3(signature = (handle, agent_name, provider_handle, instructions=None))]
pub fn team_add_agent(
    handle: u32,
    agent_name: String,
    provider_handle: u32,
    instructions: Option<String>,
) -> PyResult<()> {
    TEAMS.with_mut(handle, |state| {
        state.agents.push(TeamAgentDef {
            name: agent_name,
            provider_handle,
            instructions,
        });
        Ok(())
    })
}

#[pyfunction]
pub fn team_set_strategy(handle: u32, strategy: String) -> PyResult<()> {
    TEAMS.with_mut(handle, |state| {
        match strategy.as_str() {
            "sequential" | "parallel" => {
                state.strategy = strategy;
                Ok(())
            }
            _ => Err(py_err(format!(
                "Unknown strategy: {strategy}. Use 'sequential' or 'parallel'"
            ))),
        }
    })
}

#[pyfunction]
pub fn team_run<'py>(
    py: Python<'py>,
    handle: u32,
    messages_json: String,
) -> PyResult<Bound<'py, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let state = TEAMS.get(handle, |s| s.clone())?;

        let strategy = match state.strategy.as_str() {
            "parallel" => team::Strategy::Parallel,
            _ => team::Strategy::Sequential,
        };

        let mut builder = team::Team::builder(&state.name).strategy(strategy);

        for agent_def in &state.agents {
            let provider = get_provider(agent_def.provider_handle)?;
            let mut ab = RustAgent::builder(&agent_def.name, provider);
            if let Some(ref instr) = agent_def.instructions {
                ab = ab.instructions(instr.clone());
            }
            builder = builder.agent(ab.build());
        }

        let team_instance = builder.build();

        let messages: Vec<RustMessage> = serde_json::from_str(&messages_json)
            .map_err(|e| py_err(format!("Invalid messages JSON: {e}")))?;

        let output = team_instance
            .run(messages)
            .await
            .map_err(|e| py_err(format!("Team run error: {e}")))?;

        let results: Vec<serde_json::Value> = output
            .results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "text": r.text,
                    "steps": r.steps,
                    "inputTokens": r.usage.input_tokens,
                    "outputTokens": r.usage.output_tokens,
                })
            })
            .collect();

        serde_json::to_string(&serde_json::json!({
            "finalText": output.final_text,
            "results": results,
        }))
        .map_err(|e| py_err(format!("Serialize error: {e}")))
    })
}

#[pyfunction]
pub fn destroy_team(handle: u32) -> PyResult<()> {
    TEAMS.remove(handle)
}
