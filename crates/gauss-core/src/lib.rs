//! # Gauss Core
//!
//! High-performance AI agent engine in Rust.
//! Provides the foundation for building AI agents with tool use,
//! streaming, structured output, and multi-provider support.

pub mod agent;
pub mod error;
pub mod message;
pub mod provider;
pub mod streaming;
pub mod team;
pub mod tool;
pub mod workflow;

/// Platform-aware shared pointer: `Arc` on native, `Rc` on WASM.
#[cfg(not(target_arch = "wasm32"))]
pub type Shared<T> = std::sync::Arc<T>;
#[cfg(target_arch = "wasm32")]
pub type Shared<T> = std::rc::Rc<T>;

pub use agent::{Agent, AgentBuilder, AgentOutput};
pub use error::GaussError;
pub use message::{Content, Message, Role};
pub use provider::{GenerateOptions, Provider, ProviderConfig};
pub use streaming::StreamEvent;
pub use team::{Team, TeamBuilder, TeamOutput};
pub use tool::{Tool, ToolChoice};
pub use workflow::{Workflow, WorkflowBuilder};
