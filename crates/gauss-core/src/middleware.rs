//! Middleware system — composable, priority-ordered hooks for agent execution.
//!
//! Mirrors the gauss-flow TypeScript middleware with before/after agent
//! and before/after tool call hooks, priority ordering, and error handling.

use crate::error;
use crate::message::Message;
use crate::tool::Tool;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Priority
// ---------------------------------------------------------------------------

/// Execution priority — lower values execute first.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
pub enum MiddlewarePriority {
    /// Security, auth, rate limiting.
    First = 0,
    /// Validation, input transformation.
    Early = 250,
    /// Default.
    #[default]
    Normal = 500,
    /// Logging, metrics.
    Late = 750,
    /// Cleanup, final telemetry.
    Last = 1000,
}

// ---------------------------------------------------------------------------
// Context & Hook Types
// ---------------------------------------------------------------------------

/// Shared context passed through the middleware chain.
#[derive(Debug, Clone)]
pub struct MiddlewareContext {
    pub session_id: String,
    pub agent_name: Option<String>,
    pub timestamp: u64,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Parameters for `before_agent` hook.
#[derive(Debug, Clone)]
pub struct BeforeAgentParams {
    pub messages: Vec<Message>,
    pub instructions: Option<String>,
    pub tools: Vec<Tool>,
}

/// Result of `before_agent` hook.
#[derive(Debug, Clone, Default)]
pub struct BeforeAgentResult {
    /// Modified messages (None = use original).
    pub messages: Option<Vec<Message>>,
    /// Modified instructions.
    pub instructions: Option<String>,
    /// Additional tools to inject.
    pub tools: Option<Vec<Tool>>,
    /// If true, skip agent execution and use `early_result`.
    pub abort: bool,
    /// Text to return when aborted.
    pub early_result: Option<String>,
}

/// Parameters for `after_agent` hook.
#[derive(Debug, Clone)]
pub struct AfterAgentParams {
    pub messages: Vec<Message>,
    pub result_text: String,
    pub session_id: String,
}

/// Result of `after_agent` hook.
#[derive(Debug, Clone, Default)]
pub struct AfterAgentResult {
    pub text: Option<String>,
}

/// Parameters for `before_tool` hook.
#[derive(Debug, Clone)]
pub struct BeforeToolParams {
    pub tool_name: String,
    pub args: serde_json::Value,
    pub step_index: usize,
}

/// Result of `before_tool` hook.
#[derive(Debug, Clone, Default)]
pub struct BeforeToolResult {
    pub args: Option<serde_json::Value>,
    pub skip: bool,
    pub mock_result: Option<serde_json::Value>,
}

/// Parameters for `after_tool` hook.
#[derive(Debug, Clone)]
pub struct AfterToolParams {
    pub tool_name: String,
    pub args: serde_json::Value,
    pub result: serde_json::Value,
    pub step_index: usize,
    pub duration_ms: u64,
}

/// Result of `after_tool` hook.
#[derive(Debug, Clone, Default)]
pub struct AfterToolResult {
    pub result: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Middleware Trait
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
pub trait Middleware: Send + Sync {
    fn name(&self) -> &str;
    fn priority(&self) -> MiddlewarePriority {
        MiddlewarePriority::Normal
    }

    async fn before_agent(
        &self,
        _ctx: &mut MiddlewareContext,
        _params: &BeforeAgentParams,
    ) -> error::Result<Option<BeforeAgentResult>> {
        Ok(None)
    }

    async fn after_agent(
        &self,
        _ctx: &mut MiddlewareContext,
        _params: &AfterAgentParams,
    ) -> error::Result<Option<AfterAgentResult>> {
        Ok(None)
    }

    async fn before_tool(
        &self,
        _ctx: &mut MiddlewareContext,
        _params: &BeforeToolParams,
    ) -> error::Result<Option<BeforeToolResult>> {
        Ok(None)
    }

    async fn after_tool(
        &self,
        _ctx: &mut MiddlewareContext,
        _params: &AfterToolParams,
    ) -> error::Result<Option<AfterToolResult>> {
        Ok(None)
    }

    async fn setup(&self, _ctx: &mut MiddlewareContext) -> error::Result<()> {
        Ok(())
    }

    async fn teardown(&self, _ctx: &mut MiddlewareContext) -> error::Result<()> {
        Ok(())
    }
}

#[cfg(target_arch = "wasm32")]
#[async_trait(?Send)]
pub trait Middleware {
    fn name(&self) -> &str;
    fn priority(&self) -> MiddlewarePriority {
        MiddlewarePriority::Normal
    }

    async fn before_agent(
        &self,
        _ctx: &mut MiddlewareContext,
        _params: &BeforeAgentParams,
    ) -> error::Result<Option<BeforeAgentResult>> {
        Ok(None)
    }

    async fn after_agent(
        &self,
        _ctx: &mut MiddlewareContext,
        _params: &AfterAgentParams,
    ) -> error::Result<Option<AfterAgentResult>> {
        Ok(None)
    }

    async fn before_tool(
        &self,
        _ctx: &mut MiddlewareContext,
        _params: &BeforeToolParams,
    ) -> error::Result<Option<BeforeToolResult>> {
        Ok(None)
    }

    async fn after_tool(
        &self,
        _ctx: &mut MiddlewareContext,
        _params: &AfterToolParams,
    ) -> error::Result<Option<AfterToolResult>> {
        Ok(None)
    }

    async fn setup(&self, _ctx: &mut MiddlewareContext) -> error::Result<()> {
        Ok(())
    }

    async fn teardown(&self, _ctx: &mut MiddlewareContext) -> error::Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Middleware Chain
// ---------------------------------------------------------------------------

/// Ordered middleware execution pipeline.
pub struct MiddlewareChain {
    middlewares: Vec<crate::Shared<dyn Middleware>>,
}

impl Default for MiddlewareChain {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for MiddlewareChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MiddlewareChain")
            .field("count", &self.middlewares.len())
            .finish()
    }
}

impl MiddlewareChain {
    pub fn new() -> Self {
        Self {
            middlewares: Vec::new(),
        }
    }

    /// Add a middleware, maintaining priority order.
    pub fn use_middleware(&mut self, mw: crate::Shared<dyn Middleware>) {
        self.middlewares.push(mw);
        self.middlewares.sort_by_key(|m| m.priority() as u16);
    }

    /// Remove middleware by name.
    pub fn remove(&mut self, name: &str) -> bool {
        let before = self.middlewares.len();
        self.middlewares.retain(|m| m.name() != name);
        self.middlewares.len() < before
    }

    /// List registered middleware names.
    pub fn list(&self) -> Vec<&str> {
        self.middlewares.iter().map(|m| m.name()).collect()
    }

    /// Run all `before_agent` hooks in priority order.
    pub async fn run_before_agent(
        &self,
        ctx: &mut MiddlewareContext,
        mut params: BeforeAgentParams,
    ) -> error::Result<(BeforeAgentParams, bool, Option<String>)> {
        for mw in &self.middlewares {
            if let Some(result) = mw.before_agent(ctx, &params).await? {
                if let Some(msgs) = result.messages {
                    params.messages = msgs;
                }
                if let Some(instr) = result.instructions {
                    params.instructions = Some(instr);
                }
                if let Some(tools) = result.tools {
                    params.tools.extend(tools);
                }
                if result.abort {
                    return Ok((params, true, result.early_result));
                }
            }
        }
        Ok((params, false, None))
    }

    /// Run all `after_agent` hooks in reverse priority order.
    pub async fn run_after_agent(
        &self,
        ctx: &mut MiddlewareContext,
        mut params: AfterAgentParams,
    ) -> error::Result<AfterAgentParams> {
        for mw in self.middlewares.iter().rev() {
            if let Some(result) = mw.after_agent(ctx, &params).await?
                && let Some(text) = result.text
            {
                params.result_text = text;
            }
        }
        Ok(params)
    }

    /// Run all `before_tool` hooks in priority order.
    pub async fn run_before_tool(
        &self,
        ctx: &mut MiddlewareContext,
        mut params: BeforeToolParams,
    ) -> error::Result<(BeforeToolParams, bool, Option<serde_json::Value>)> {
        for mw in &self.middlewares {
            if let Some(result) = mw.before_tool(ctx, &params).await? {
                if let Some(args) = result.args {
                    params.args = args;
                }
                if result.skip {
                    return Ok((params, true, result.mock_result));
                }
            }
        }
        Ok((params, false, None))
    }

    /// Run all `after_tool` hooks in reverse priority order.
    pub async fn run_after_tool(
        &self,
        ctx: &mut MiddlewareContext,
        mut params: AfterToolParams,
    ) -> error::Result<AfterToolParams> {
        for mw in self.middlewares.iter().rev() {
            if let Some(result) = mw.after_tool(ctx, &params).await?
                && let Some(r) = result.result
            {
                params.result = r;
            }
        }
        Ok(params)
    }

    /// Initialize all middleware.
    pub async fn setup(&self, ctx: &mut MiddlewareContext) -> error::Result<()> {
        for mw in &self.middlewares {
            mw.setup(ctx).await?;
        }
        Ok(())
    }

    /// Teardown all middleware.
    pub async fn teardown(&self, ctx: &mut MiddlewareContext) -> error::Result<()> {
        for mw in self.middlewares.iter().rev() {
            mw.teardown(ctx).await?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Built-in: Logging Middleware
// ---------------------------------------------------------------------------

/// Simple logging middleware using the `tracing` crate.
#[derive(Debug)]
pub struct LoggingMiddleware;

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl Middleware for LoggingMiddleware {
    fn name(&self) -> &str {
        "logging"
    }

    fn priority(&self) -> MiddlewarePriority {
        MiddlewarePriority::Late
    }

    async fn before_agent(
        &self,
        ctx: &mut MiddlewareContext,
        params: &BeforeAgentParams,
    ) -> error::Result<Option<BeforeAgentResult>> {
        tracing::info!(
            session_id = %ctx.session_id,
            message_count = params.messages.len(),
            tool_count = params.tools.len(),
            "Agent execution starting"
        );
        Ok(None)
    }

    async fn after_agent(
        &self,
        ctx: &mut MiddlewareContext,
        params: &AfterAgentParams,
    ) -> error::Result<Option<AfterAgentResult>> {
        tracing::info!(
            session_id = %ctx.session_id,
            result_len = params.result_text.len(),
            "Agent execution completed"
        );
        Ok(None)
    }

    async fn before_tool(
        &self,
        ctx: &mut MiddlewareContext,
        params: &BeforeToolParams,
    ) -> error::Result<Option<BeforeToolResult>> {
        tracing::info!(
            session_id = %ctx.session_id,
            tool = %params.tool_name,
            step = params.step_index,
            "Tool call starting"
        );
        Ok(None)
    }

    async fn after_tool(
        &self,
        ctx: &mut MiddlewareContext,
        params: &AfterToolParams,
    ) -> error::Result<Option<AfterToolResult>> {
        tracing::info!(
            session_id = %ctx.session_id,
            tool = %params.tool_name,
            duration_ms = params.duration_ms,
            "Tool call completed"
        );
        Ok(None)
    }
}

// ---------------------------------------------------------------------------
// Built-in: Caching Middleware
// ---------------------------------------------------------------------------

/// LLM response caching middleware with TTL.
#[derive(Debug)]
pub struct CachingMiddleware {
    cache: std::sync::Mutex<HashMap<String, CacheEntry>>,
    ttl_ms: u64,
}

#[derive(Debug)]
struct CacheEntry {
    value: String,
    expires_at: u64,
}

impl CachingMiddleware {
    pub fn new(ttl_ms: u64) -> Self {
        Self {
            cache: std::sync::Mutex::new(HashMap::new()),
            ttl_ms,
        }
    }

    fn now_millis() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    fn cache_key(params: &BeforeAgentParams) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        for msg in &params.messages {
            format!("{:?}", msg).hash(&mut hasher);
        }
        format!("{:x}", hasher.finish())
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl Middleware for CachingMiddleware {
    fn name(&self) -> &str {
        "caching"
    }

    fn priority(&self) -> MiddlewarePriority {
        MiddlewarePriority::Early
    }

    async fn before_agent(
        &self,
        _ctx: &mut MiddlewareContext,
        params: &BeforeAgentParams,
    ) -> error::Result<Option<BeforeAgentResult>> {
        let key = Self::cache_key(params);
        let now = Self::now_millis();
        let cache = self
            .cache
            .lock()
            .map_err(|e| error::GaussError::internal(e.to_string()))?;

        if let Some(entry) = cache.get(&key).filter(|entry| entry.expires_at > now) {
            return Ok(Some(BeforeAgentResult {
                abort: true,
                early_result: Some(entry.value.clone()),
                ..Default::default()
            }));
        }
        Ok(None)
    }

    async fn after_agent(
        &self,
        _ctx: &mut MiddlewareContext,
        params: &AfterAgentParams,
    ) -> error::Result<Option<AfterAgentResult>> {
        let key = Self::cache_key(&BeforeAgentParams {
            messages: params.messages.clone(),
            instructions: None,
            tools: Vec::new(),
        });
        let now = Self::now_millis();
        let mut cache = self
            .cache
            .lock()
            .map_err(|e| error::GaussError::internal(e.to_string()))?;
        cache.insert(
            key,
            CacheEntry {
                value: params.result_text.clone(),
                expires_at: now + self.ttl_ms,
            },
        );
        Ok(None)
    }
}
