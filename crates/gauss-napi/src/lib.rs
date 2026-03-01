#[macro_use]
extern crate napi_derive;

pub mod registry;
pub mod types;
pub mod provider;
pub mod agent;
pub mod code_exec;
pub mod memory;
pub mod mcp;
pub mod network;
pub mod middleware;
pub mod hitl;
pub mod eval;
pub mod orchestration;
pub mod plugin;
pub mod config;

/// Gauss Core version.
#[napi]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
