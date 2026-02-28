//! Plugin system — formal lifecycle, event bus, and registry for extensibility.
//!
//! Provides a composable plugin architecture with typed event bus,
//! dependency-aware registry, and lifecycle management.

use crate::error::{self, GaussError};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// GaussEvent — typed events for the bus
// ---------------------------------------------------------------------------

/// Events published through the event bus.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GaussEvent {
    /// Agent execution started.
    AgentStart {
        agent_name: String,
        session_id: String,
    },
    /// Agent execution finished.
    AgentFinish {
        agent_name: String,
        session_id: String,
        result_text: String,
    },
    /// Tool call started.
    ToolCallStart {
        tool_name: String,
        arguments: serde_json::Value,
    },
    /// Tool call completed.
    ToolCallFinish {
        tool_name: String,
        result: serde_json::Value,
        duration_ms: u64,
        is_error: bool,
    },
    /// An error occurred.
    Error { source: String, message: String },
    /// Custom user-defined event.
    Custom {
        name: String,
        data: serde_json::Value,
    },
}

impl GaussEvent {
    /// Get the event type name.
    pub fn event_type(&self) -> &str {
        match self {
            Self::AgentStart { .. } => "agent_start",
            Self::AgentFinish { .. } => "agent_finish",
            Self::ToolCallStart { .. } => "tool_call_start",
            Self::ToolCallFinish { .. } => "tool_call_finish",
            Self::Error { .. } => "error",
            Self::Custom { name, .. } => name,
        }
    }
}

// ---------------------------------------------------------------------------
// EventBus
// ---------------------------------------------------------------------------

/// Handler function type for event bus.
#[cfg(not(target_arch = "wasm32"))]
pub type EventHandler = Box<dyn Fn(&GaussEvent) + Send + Sync>;
#[cfg(target_arch = "wasm32")]
pub type EventHandler = Box<dyn Fn(&GaussEvent)>;

/// Publish/subscribe event bus with typed events.
pub struct EventBus {
    /// Handlers keyed by event type pattern. "*" matches all events.
    handlers: HashMap<String, Vec<(String, EventHandler)>>,
    next_id: u32,
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for EventBus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let total: usize = self.handlers.values().map(|v| v.len()).sum();
        f.debug_struct("EventBus")
            .field("handler_count", &total)
            .field("event_types", &self.handlers.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl EventBus {
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
            next_id: 0,
        }
    }

    /// Subscribe to a specific event type. Use "*" for all events.
    /// Returns a subscription ID for unsubscribing.
    pub fn subscribe(&mut self, event_type: impl Into<String>, handler: EventHandler) -> String {
        let id = format!("sub_{}", self.next_id);
        self.next_id += 1;
        self.handlers
            .entry(event_type.into())
            .or_default()
            .push((id.clone(), handler));
        id
    }

    /// Unsubscribe by subscription ID.
    pub fn unsubscribe(&mut self, subscription_id: &str) -> bool {
        let mut found = false;
        for handlers in self.handlers.values_mut() {
            let before = handlers.len();
            handlers.retain(|(id, _)| id != subscription_id);
            if handlers.len() < before {
                found = true;
            }
        }
        found
    }

    /// Publish an event to all matching handlers.
    pub fn publish(&self, event: &GaussEvent) {
        let event_type = event.event_type();

        // Specific handlers
        if let Some(handlers) = self.handlers.get(event_type) {
            for (_, handler) in handlers {
                handler(event);
            }
        }

        // Wildcard handlers
        if let Some(handlers) = self.handlers.get("*") {
            for (_, handler) in handlers {
                handler(event);
            }
        }
    }

    /// Get the number of subscriptions.
    pub fn subscription_count(&self) -> usize {
        self.handlers.values().map(|v| v.len()).sum()
    }
}

// ---------------------------------------------------------------------------
// PluginContext
// ---------------------------------------------------------------------------

/// Context provided to plugins during initialization.
pub struct PluginContext {
    /// Shared configuration values.
    pub config: HashMap<String, serde_json::Value>,
    /// Shared state that plugins can read/write.
    pub state: HashMap<String, serde_json::Value>,
}

impl Default for PluginContext {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginContext {
    pub fn new() -> Self {
        Self {
            config: HashMap::new(),
            state: HashMap::new(),
        }
    }

    pub fn with_config(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.config.insert(key.into(), value);
        self
    }
}

// ---------------------------------------------------------------------------
// Plugin Trait
// ---------------------------------------------------------------------------

/// A plugin extends Gauss with custom behavior via the event bus.
#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
pub trait Plugin: Send + Sync {
    /// Unique plugin name.
    fn name(&self) -> &str;

    /// Plugin version (semver).
    fn version(&self) -> &str;

    /// Optional list of plugin names this plugin depends on.
    fn dependencies(&self) -> Vec<&str> {
        Vec::new()
    }

    /// Initialize the plugin. Register event handlers here.
    async fn init(&self, ctx: &mut PluginContext, bus: &mut EventBus) -> error::Result<()>;

    /// Shutdown the plugin. Clean up resources.
    async fn shutdown(&self, _ctx: &mut PluginContext) -> error::Result<()> {
        Ok(())
    }
}

#[cfg(target_arch = "wasm32")]
#[async_trait(?Send)]
pub trait Plugin {
    fn name(&self) -> &str;
    fn version(&self) -> &str;

    fn dependencies(&self) -> Vec<&str> {
        Vec::new()
    }

    async fn init(&self, ctx: &mut PluginContext, bus: &mut EventBus) -> error::Result<()>;

    async fn shutdown(&self, _ctx: &mut PluginContext) -> error::Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PluginRegistry
// ---------------------------------------------------------------------------

/// Manages plugin lifecycle and dependency ordering.
pub struct PluginRegistry {
    plugins: Vec<crate::Shared<dyn Plugin>>,
    initialized: Vec<String>,
    pub bus: EventBus,
    pub ctx: PluginContext,
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginRegistry {
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
            initialized: Vec::new(),
            bus: EventBus::new(),
            ctx: PluginContext::new(),
        }
    }

    /// Register a plugin. Does NOT initialize it yet.
    pub fn register(&mut self, plugin: crate::Shared<dyn Plugin>) {
        self.plugins.push(plugin);
    }

    /// List registered plugin names.
    pub fn list(&self) -> Vec<&str> {
        self.plugins.iter().map(|p| p.name()).collect()
    }

    /// Initialize all plugins in dependency order.
    pub async fn init_all(&mut self) -> error::Result<()> {
        // Topological sort by dependencies
        let sorted = self.topological_sort()?;

        for plugin in &sorted {
            if !self.initialized.contains(&plugin.name().to_string()) {
                plugin.init(&mut self.ctx, &mut self.bus).await?;
                self.initialized.push(plugin.name().to_string());
            }
        }

        Ok(())
    }

    /// Shutdown all plugins in reverse initialization order.
    pub async fn shutdown_all(&mut self) -> error::Result<()> {
        for name in self.initialized.iter().rev() {
            if let Some(plugin) = self.plugins.iter().find(|p| p.name() == name) {
                plugin.shutdown(&mut self.ctx).await?;
            }
        }
        self.initialized.clear();
        Ok(())
    }

    /// Publish an event to the bus.
    pub fn emit(&self, event: &GaussEvent) {
        self.bus.publish(event);
    }

    fn topological_sort(&self) -> error::Result<Vec<crate::Shared<dyn Plugin>>> {
        let mut sorted: Vec<crate::Shared<dyn Plugin>> = Vec::new();
        let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut in_progress: std::collections::HashSet<String> = std::collections::HashSet::new();

        let plugin_map: HashMap<&str, &crate::Shared<dyn Plugin>> =
            self.plugins.iter().map(|p| (p.name(), p)).collect();

        fn visit(
            name: &str,
            plugin_map: &HashMap<&str, &crate::Shared<dyn Plugin>>,
            visited: &mut std::collections::HashSet<String>,
            in_progress: &mut std::collections::HashSet<String>,
            sorted: &mut Vec<crate::Shared<dyn Plugin>>,
        ) -> error::Result<()> {
            if visited.contains(name) {
                return Ok(());
            }
            if in_progress.contains(name) {
                return Err(GaussError::internal(format!(
                    "Circular plugin dependency: {name}"
                )));
            }

            in_progress.insert(name.to_string());

            if let Some(plugin) = plugin_map.get(name) {
                for dep in plugin.dependencies() {
                    visit(dep, plugin_map, visited, in_progress, sorted)?;
                }
                sorted.push((*plugin).clone());
            }

            in_progress.remove(name);
            visited.insert(name.to_string());
            Ok(())
        }

        for plugin in &self.plugins {
            visit(
                plugin.name(),
                &plugin_map,
                &mut visited,
                &mut in_progress,
                &mut sorted,
            )?;
        }

        Ok(sorted)
    }
}

// ---------------------------------------------------------------------------
// Built-in: TelemetryPlugin
// ---------------------------------------------------------------------------

/// Plugin that logs all events via tracing.
pub struct TelemetryPlugin;

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl Plugin for TelemetryPlugin {
    fn name(&self) -> &str {
        "telemetry"
    }

    fn version(&self) -> &str {
        "0.1.0"
    }

    async fn init(&self, _ctx: &mut PluginContext, bus: &mut EventBus) -> error::Result<()> {
        bus.subscribe(
            "*",
            Box::new(|event| {
                tracing::info!(event_type = event.event_type(), "Plugin event");
            }),
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Built-in: MemoryPlugin
// ---------------------------------------------------------------------------

/// Plugin that stores agent interactions in shared state for memory.
pub struct MemoryPlugin;

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl Plugin for MemoryPlugin {
    fn name(&self) -> &str {
        "memory"
    }

    fn version(&self) -> &str {
        "0.1.0"
    }

    async fn init(&self, ctx: &mut PluginContext, bus: &mut EventBus) -> error::Result<()> {
        // Initialize conversation log in context state
        ctx.state.insert(
            "memory:conversations".to_string(),
            serde_json::Value::Array(Vec::new()),
        );

        // NOTE: The handler captures nothing from ctx — it receives the event.
        // Real memory storage would need Arc<Mutex<>> or channel-based approach.
        bus.subscribe(
            "agent_finish",
            Box::new(|event| {
                if let GaussEvent::AgentFinish {
                    agent_name,
                    session_id,
                    result_text,
                } = event
                {
                    tracing::debug!(
                        agent = %agent_name,
                        session = %session_id,
                        len = result_text.len(),
                        "MemoryPlugin: storing conversation"
                    );
                }
            }),
        );

        Ok(())
    }
}
