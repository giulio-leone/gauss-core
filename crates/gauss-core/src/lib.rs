//! # Gauss Core
//!
//! High-performance AI agent engine in Rust.
//! Provides the foundation for building AI agents with tool use,
//! streaming, structured output, multi-provider support, memory,
//! RAG, MCP, middleware, observability, and multi-agent networks.

pub mod agent;
#[cfg(not(target_arch = "wasm32"))]
pub mod code_execution;
pub mod config;
pub mod context;
pub mod error;
pub mod eval;
pub mod graph;
pub mod guardrail;
pub mod hitl;
pub mod mcp;
pub mod memory;
pub mod message;
pub mod middleware;
pub mod network;
pub mod patterns;
pub mod plugin;
pub mod provider;
pub mod rag;
pub mod resilience;
pub mod stream_transform;
pub mod streaming;
pub mod team;
pub mod telemetry;
pub mod tool;
pub mod workflow;

/// Platform-aware shared pointer: `Arc` on native, `Rc` on WASM.
#[cfg(not(target_arch = "wasm32"))]
pub type Shared<T> = std::sync::Arc<T>;
#[cfg(target_arch = "wasm32")]
pub type Shared<T> = std::rc::Rc<T>;

pub use agent::{Agent, AgentBuilder, AgentOutput};
#[cfg(not(target_arch = "wasm32"))]
pub use code_execution::{
    BashRuntime, CodeRuntime, ExecutionResult, JavaScriptRuntime, PythonRuntime, RuntimeConfig,
    code_execution_tool,
};
pub use error::GaussError;
pub use graph::{ConsensusStrategy, Graph, GraphBuilder, GraphResult};
pub use message::{Citation, Content, Message, Role};
pub use provider::{GenerateOptions, Provider, ProviderCapabilities, ProviderConfig};
pub use streaming::StreamEvent;
pub use team::{Team, TeamBuilder, TeamOutput};
pub use tool::{Tool, ToolChoice};
pub use workflow::{Workflow, WorkflowBuilder};
