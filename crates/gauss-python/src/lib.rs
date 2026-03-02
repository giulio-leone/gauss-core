pub mod agent;
pub mod code_exec;
pub mod config;
pub mod eval;
pub mod hitl;
pub mod mcp;
pub mod memory;
pub mod middleware;
pub mod network;
pub mod orchestration;
pub mod plugin;
pub mod provider;
pub mod registry;
pub mod types;

use pyo3::prelude::*;

/// Gauss Core version.
#[pyfunction]
fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Gauss Core Python module.
#[pymodule]
#[pyo3(name = "_native")]
fn gauss_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(provider::create_provider, m)?)?;
    m.add_function(wrap_pyfunction!(provider::destroy_provider, m)?)?;
    m.add_function(wrap_pyfunction!(provider::get_provider_capabilities, m)?)?;
    m.add_function(wrap_pyfunction!(provider::estimate_cost, m)?)?;
    m.add_function(wrap_pyfunction!(code_exec::generate, m)?)?;
    m.add_function(wrap_pyfunction!(code_exec::generate_with_tools, m)?)?;
    m.add_function(wrap_pyfunction!(code_exec::stream_generate, m)?)?;
    m.add_function(wrap_pyfunction!(agent::agent_run, m)?)?;
    m.add_function(wrap_pyfunction!(agent::agent_run_with_tool_executor, m)?)?;
    m.add_function(wrap_pyfunction!(agent::agent_stream, m)?)?;
    // Memory
    m.add_function(wrap_pyfunction!(memory::create_memory, m)?)?;
    m.add_function(wrap_pyfunction!(memory::memory_store, m)?)?;
    m.add_function(wrap_pyfunction!(memory::memory_recall, m)?)?;
    m.add_function(wrap_pyfunction!(memory::memory_clear, m)?)?;
    m.add_function(wrap_pyfunction!(memory::destroy_memory, m)?)?;
    // Context
    m.add_function(wrap_pyfunction!(memory::count_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(memory::count_tokens_for_model, m)?)?;
    m.add_function(wrap_pyfunction!(memory::count_message_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(memory::get_context_window_size, m)?)?;
    // RAG
    m.add_function(wrap_pyfunction!(memory::create_vector_store, m)?)?;
    m.add_function(wrap_pyfunction!(memory::vector_store_upsert, m)?)?;
    m.add_function(wrap_pyfunction!(memory::vector_store_search, m)?)?;
    m.add_function(wrap_pyfunction!(memory::destroy_vector_store, m)?)?;
    m.add_function(wrap_pyfunction!(memory::cosine_similarity, m)?)?;
    // MCP
    m.add_function(wrap_pyfunction!(mcp::create_mcp_server, m)?)?;
    m.add_function(wrap_pyfunction!(mcp::mcp_server_add_tool, m)?)?;
    m.add_function(wrap_pyfunction!(mcp::mcp_server_add_resource, m)?)?;
    m.add_function(wrap_pyfunction!(mcp::mcp_server_add_prompt, m)?)?;
    m.add_function(wrap_pyfunction!(mcp::mcp_server_handle, m)?)?;
    m.add_function(wrap_pyfunction!(mcp::destroy_mcp_server, m)?)?;
    // Network
    m.add_function(wrap_pyfunction!(network::create_network, m)?)?;
    m.add_function(wrap_pyfunction!(network::network_add_agent, m)?)?;
    m.add_function(wrap_pyfunction!(network::network_set_supervisor, m)?)?;
    m.add_function(wrap_pyfunction!(network::network_delegate, m)?)?;
    m.add_function(wrap_pyfunction!(network::destroy_network, m)?)?;
    // HITL
    m.add_function(wrap_pyfunction!(hitl::create_approval_manager, m)?)?;
    m.add_function(wrap_pyfunction!(hitl::approval_request, m)?)?;
    m.add_function(wrap_pyfunction!(hitl::approval_approve, m)?)?;
    m.add_function(wrap_pyfunction!(hitl::approval_deny, m)?)?;
    m.add_function(wrap_pyfunction!(hitl::approval_list_pending, m)?)?;
    m.add_function(wrap_pyfunction!(hitl::destroy_approval_manager, m)?)?;
    // Checkpoint
    m.add_function(wrap_pyfunction!(hitl::create_checkpoint_store, m)?)?;
    m.add_function(wrap_pyfunction!(hitl::checkpoint_save, m)?)?;
    m.add_function(wrap_pyfunction!(hitl::checkpoint_load, m)?)?;
    m.add_function(wrap_pyfunction!(hitl::destroy_checkpoint_store, m)?)?;
    // Eval
    m.add_function(wrap_pyfunction!(eval::create_eval_runner, m)?)?;
    m.add_function(wrap_pyfunction!(eval::eval_add_scorer, m)?)?;
    m.add_function(wrap_pyfunction!(eval::load_dataset_jsonl, m)?)?;
    m.add_function(wrap_pyfunction!(eval::load_dataset_json, m)?)?;
    m.add_function(wrap_pyfunction!(eval::destroy_eval_runner, m)?)?;
    // Telemetry
    m.add_function(wrap_pyfunction!(eval::create_telemetry, m)?)?;
    m.add_function(wrap_pyfunction!(eval::telemetry_record_span, m)?)?;
    m.add_function(wrap_pyfunction!(eval::telemetry_export_spans, m)?)?;
    m.add_function(wrap_pyfunction!(eval::telemetry_export_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(eval::telemetry_clear, m)?)?;
    m.add_function(wrap_pyfunction!(eval::destroy_telemetry, m)?)?;
    // Guardrails
    m.add_function(wrap_pyfunction!(middleware::create_guardrail_chain, m)?)?;
    m.add_function(wrap_pyfunction!(
        middleware::guardrail_chain_add_content_moderation,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        middleware::guardrail_chain_add_pii_detection,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        middleware::guardrail_chain_add_token_limit,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        middleware::guardrail_chain_add_regex_filter,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(middleware::guardrail_chain_add_schema, m)?)?;
    m.add_function(wrap_pyfunction!(middleware::guardrail_chain_list, m)?)?;
    m.add_function(wrap_pyfunction!(middleware::destroy_guardrail_chain, m)?)?;
    // Resilience
    m.add_function(wrap_pyfunction!(middleware::create_fallback_provider, m)?)?;
    m.add_function(wrap_pyfunction!(middleware::create_circuit_breaker, m)?)?;
    m.add_function(wrap_pyfunction!(middleware::create_resilient_provider, m)?)?;
    // Stream Transform
    m.add_function(wrap_pyfunction!(middleware::py_parse_partial_json, m)?)?;
    // Plugin
    m.add_function(wrap_pyfunction!(plugin::create_plugin_registry, m)?)?;
    m.add_function(wrap_pyfunction!(plugin::plugin_registry_add_telemetry, m)?)?;
    m.add_function(wrap_pyfunction!(plugin::plugin_registry_add_memory, m)?)?;
    m.add_function(wrap_pyfunction!(plugin::plugin_registry_list, m)?)?;
    m.add_function(wrap_pyfunction!(plugin::plugin_registry_emit, m)?)?;
    m.add_function(wrap_pyfunction!(plugin::destroy_plugin_registry, m)?)?;
    // Patterns
    m.add_function(wrap_pyfunction!(plugin::create_tool_validator, m)?)?;
    m.add_function(wrap_pyfunction!(plugin::tool_validator_validate, m)?)?;
    m.add_function(wrap_pyfunction!(plugin::destroy_tool_validator, m)?)?;
    // Config
    m.add_function(wrap_pyfunction!(config::agent_config_from_json, m)?)?;
    m.add_function(wrap_pyfunction!(config::agent_config_resolve_env, m)?)?;
    // Graph
    m.add_function(wrap_pyfunction!(orchestration::create_graph, m)?)?;
    m.add_function(wrap_pyfunction!(orchestration::graph_add_node, m)?)?;
    m.add_function(wrap_pyfunction!(orchestration::graph_add_edge, m)?)?;
    m.add_function(wrap_pyfunction!(orchestration::graph_add_fork_node, m)?)?;
    m.add_function(wrap_pyfunction!(orchestration::graph_run, m)?)?;
    m.add_function(wrap_pyfunction!(orchestration::destroy_graph, m)?)?;
    // Workflow
    m.add_function(wrap_pyfunction!(orchestration::create_workflow, m)?)?;
    m.add_function(wrap_pyfunction!(orchestration::workflow_add_step, m)?)?;
    m.add_function(wrap_pyfunction!(orchestration::workflow_add_dependency, m)?)?;
    m.add_function(wrap_pyfunction!(orchestration::workflow_run, m)?)?;
    m.add_function(wrap_pyfunction!(orchestration::destroy_workflow, m)?)?;
    // Middleware
    m.add_function(wrap_pyfunction!(middleware::create_middleware_chain, m)?)?;
    m.add_function(wrap_pyfunction!(middleware::middleware_use_logging, m)?)?;
    m.add_function(wrap_pyfunction!(middleware::middleware_use_caching, m)?)?;
    m.add_function(wrap_pyfunction!(middleware::middleware_use_rate_limit, m)?)?;
    m.add_function(wrap_pyfunction!(middleware::destroy_middleware_chain, m)?)?;
    // Additional parity functions
    m.add_function(wrap_pyfunction!(memory::memory_stats, m)?)?;
    m.add_function(wrap_pyfunction!(network::network_agent_cards, m)?)?;
    m.add_function(wrap_pyfunction!(hitl::checkpoint_load_latest, m)?)?;
    // Team
    m.add_function(wrap_pyfunction!(orchestration::create_team, m)?)?;
    m.add_function(wrap_pyfunction!(orchestration::team_add_agent, m)?)?;
    m.add_function(wrap_pyfunction!(orchestration::team_set_strategy, m)?)?;
    m.add_function(wrap_pyfunction!(orchestration::team_run, m)?)?;
    m.add_function(wrap_pyfunction!(orchestration::destroy_team, m)?)?;
    // Tool Registry
    m.add_function(wrap_pyfunction!(plugin::create_tool_registry, m)?)?;
    m.add_function(wrap_pyfunction!(plugin::tool_registry_add, m)?)?;
    m.add_function(wrap_pyfunction!(plugin::tool_registry_search, m)?)?;
    m.add_function(wrap_pyfunction!(plugin::tool_registry_by_tag, m)?)?;
    m.add_function(wrap_pyfunction!(plugin::tool_registry_list, m)?)?;
    m.add_function(wrap_pyfunction!(plugin::destroy_tool_registry, m)?)?;
    // Code Execution (PTC)
    m.add_function(wrap_pyfunction!(code_exec::execute_code, m)?)?;
    m.add_function(wrap_pyfunction!(code_exec::available_runtimes, m)?)?;
    // Image Generation
    m.add_function(wrap_pyfunction!(code_exec::generate_image, m)?)?;
    // AGENTS.MD & SKILL.MD Parsers
    m.add_function(wrap_pyfunction!(config::parse_agents_md, m)?)?;
    m.add_function(wrap_pyfunction!(config::discover_agents, m)?)?;
    m.add_function(wrap_pyfunction!(config::parse_skill_md, m)?)?;
    // A2A Protocol
    m.add_function(wrap_pyfunction!(network::a2a_discover, m)?)?;
    m.add_function(wrap_pyfunction!(network::a2a_send_message, m)?)?;
    m.add_function(wrap_pyfunction!(network::a2a_ask, m)?)?;
    m.add_function(wrap_pyfunction!(network::a2a_get_task, m)?)?;
    m.add_function(wrap_pyfunction!(network::a2a_cancel_task, m)?)?;
    Ok(())
}
