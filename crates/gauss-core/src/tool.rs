use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::error;

/// Tool choice configuration for the agent.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    Required,
    Specific {
        name: String,
    },
}

/// JSON Schema for tool parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParameters {
    #[serde(rename = "type")]
    pub schema_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl Default for ToolParameters {
    fn default() -> Self {
        Self {
            schema_type: "object".to_string(),
            properties: None,
            required: None,
            extra: serde_json::Map::new(),
        }
    }
}

/// Type alias for tool execution function.
#[cfg(not(target_arch = "wasm32"))]
pub type ToolExecuteFn = Arc<
    dyn Fn(
            serde_json::Value,
        ) -> Pin<Box<dyn Future<Output = error::Result<serde_json::Value>> + Send>>
        + Send
        + Sync,
>;

#[cfg(target_arch = "wasm32")]
pub type ToolExecuteFn = std::rc::Rc<
    dyn Fn(serde_json::Value) -> Pin<Box<dyn Future<Output = error::Result<serde_json::Value>>>>,
>;

/// A tool that can be used by an agent.
#[derive(Clone)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: ToolParameters,
    /// Optional tags for categorization and search.
    pub tags: Vec<String>,
    /// Optional usage examples.
    pub examples: Vec<ToolExample>,
    execute: Option<ToolExecuteFn>,
}

impl std::fmt::Debug for Tool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("parameters", &self.parameters)
            .field("tags", &self.tags)
            .field("examples", &self.examples)
            .field("has_execute", &self.execute.is_some())
            .finish()
    }
}

impl Tool {
    pub fn builder(name: impl Into<String>, description: impl Into<String>) -> ToolBuilder {
        ToolBuilder {
            name: name.into(),
            description: description.into(),
            parameters: ToolParameters::default(),
            tags: Vec::new(),
            examples: Vec::new(),
            execute: None,
        }
    }

    /// Execute this tool with the given arguments.
    pub async fn execute(&self, args: serde_json::Value) -> error::Result<serde_json::Value> {
        match &self.execute {
            Some(f) => f(args).await,
            None => Err(error::GaussError::tool(
                &self.name,
                "Tool has no execute function",
            )),
        }
    }

    pub fn has_execute(&self) -> bool {
        self.execute.is_some()
    }

    /// Check if this tool matches a search query (name, description, or tags).
    pub fn matches(&self, query: &str) -> bool {
        let q = query.to_lowercase();
        self.name.to_lowercase().contains(&q)
            || self.description.to_lowercase().contains(&q)
            || self.tags.iter().any(|t| t.to_lowercase().contains(&q))
    }
}

/// An example of how to use a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExample {
    pub description: String,
    pub input: serde_json::Value,
    pub expected_output: Option<serde_json::Value>,
}

pub struct ToolBuilder {
    name: String,
    description: String,
    parameters: ToolParameters,
    tags: Vec<String>,
    examples: Vec<ToolExample>,
    execute: Option<ToolExecuteFn>,
}

impl ToolBuilder {
    pub fn parameters(mut self, params: ToolParameters) -> Self {
        self.parameters = params;
        self
    }

    pub fn parameters_json(mut self, schema: serde_json::Value) -> Self {
        if let Ok(params) = serde_json::from_value(schema) {
            self.parameters = params;
        }
        self
    }

    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    pub fn tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tags.extend(tags.into_iter().map(|t| t.into()));
        self
    }

    pub fn example(mut self, example: ToolExample) -> Self {
        self.examples.push(example);
        self
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn execute<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(serde_json::Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = error::Result<serde_json::Value>> + Send + 'static,
    {
        self.execute = Some(Arc::new(move |args| Box::pin(f(args))));
        self
    }

    #[cfg(target_arch = "wasm32")]
    pub fn execute<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(serde_json::Value) -> Fut + 'static,
        Fut: Future<Output = error::Result<serde_json::Value>> + 'static,
    {
        self.execute = Some(std::rc::Rc::new(move |args| Box::pin(f(args))));
        self
    }

    pub fn build(self) -> Tool {
        Tool {
            name: self.name,
            description: self.description,
            parameters: self.parameters,
            tags: self.tags,
            examples: self.examples,
            execute: self.execute,
        }
    }
}

// ── Tool Registry ───────────────────────────────────────────────

/// A searchable registry of tools.
#[derive(Debug, Clone, Default)]
pub struct ToolRegistry {
    tools: Vec<Tool>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    /// Register a tool.
    pub fn register(&mut self, tool: Tool) {
        self.tools.push(tool);
    }

    /// Get a tool by exact name.
    pub fn get(&self, name: &str) -> Option<&Tool> {
        self.tools.iter().find(|t| t.name == name)
    }

    /// Search tools by query (matches name, description, tags).
    pub fn search(&self, query: &str) -> Vec<&Tool> {
        self.tools.iter().filter(|t| t.matches(query)).collect()
    }

    /// Search tools by tag.
    pub fn by_tag(&self, tag: &str) -> Vec<&Tool> {
        let t = tag.to_lowercase();
        self.tools
            .iter()
            .filter(|tool| tool.tags.iter().any(|tg| tg.to_lowercase() == t))
            .collect()
    }

    /// List all registered tools.
    pub fn list(&self) -> &[Tool] {
        &self.tools
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

// ── Batch Tool Execution ────────────────────────────────────────

/// Result of a single tool call in a batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchToolResult {
    pub tool_name: String,
    pub input: serde_json::Value,
    pub output: Option<serde_json::Value>,
    pub error: Option<String>,
    pub duration_ms: u64,
}

/// Options for batch tool execution.
#[derive(Debug, Clone)]
pub struct BatchOptions {
    /// Maximum number of concurrent tool calls (0 = unlimited).
    pub concurrency: usize,
    /// Whether to continue on error.
    pub continue_on_error: bool,
}

impl Default for BatchOptions {
    fn default() -> Self {
        Self {
            concurrency: 0,
            continue_on_error: true,
        }
    }
}

/// A single tool call request in a batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchToolCall {
    pub tool_name: String,
    pub input: serde_json::Value,
}

/// Execute multiple tool calls concurrently.
#[cfg(not(target_arch = "wasm32"))]
pub async fn batch_execute(
    registry: &ToolRegistry,
    calls: Vec<BatchToolCall>,
    options: &BatchOptions,
) -> Vec<BatchToolResult> {
    use tokio::sync::Semaphore;

    let sem = if options.concurrency > 0 {
        Some(Arc::new(Semaphore::new(options.concurrency)))
    } else {
        None
    };

    let mut handles = Vec::with_capacity(calls.len());

    for call in calls {
        let tool = registry.get(&call.tool_name).cloned();
        let sem = sem.clone();
        let continue_on_error = options.continue_on_error;

        handles.push(tokio::spawn(async move {
            let _permit = match &sem {
                Some(s) => Some(s.acquire().await.unwrap()),
                None => None,
            };

            let start = std::time::Instant::now();

            match tool {
                Some(t) => match t.execute(call.input.clone()).await {
                    Ok(output) => BatchToolResult {
                        tool_name: call.tool_name,
                        input: call.input,
                        output: Some(output),
                        error: None,
                        duration_ms: start.elapsed().as_millis() as u64,
                    },
                    Err(e) => {
                        if !continue_on_error {
                            // Still return the error in the result
                        }
                        BatchToolResult {
                            tool_name: call.tool_name,
                            input: call.input,
                            output: None,
                            error: Some(e.to_string()),
                            duration_ms: start.elapsed().as_millis() as u64,
                        }
                    }
                },
                None => BatchToolResult {
                    tool_name: call.tool_name.clone(),
                    input: call.input,
                    output: None,
                    error: Some(format!("Tool '{}' not found", call.tool_name)),
                    duration_ms: start.elapsed().as_millis() as u64,
                },
            }
        }));
    }

    let mut results = Vec::with_capacity(handles.len());
    for handle in handles {
        match handle.await {
            Ok(result) => results.push(result),
            Err(e) => results.push(BatchToolResult {
                tool_name: "unknown".to_string(),
                input: serde_json::Value::Null,
                output: None,
                error: Some(format!("Task join error: {e}")),
                duration_ms: 0,
            }),
        }
    }
    results
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tool(name: &str, desc: &str, tags: &[&str]) -> Tool {
        let mut builder = Tool::builder(name, desc);
        for tag in tags {
            builder = builder.tag(*tag);
        }
        builder
            .execute(|args| async move { Ok(args) })
            .build()
    }

    #[test]
    fn test_tool_builder_with_tags() {
        let tool = Tool::builder("calc", "Calculator")
            .tag("math")
            .tag("utility")
            .build();
        assert_eq!(tool.tags, vec!["math", "utility"]);
    }

    #[test]
    fn test_tool_builder_with_example() {
        let tool = Tool::builder("add", "Add numbers")
            .example(ToolExample {
                description: "Add 2 + 3".to_string(),
                input: serde_json::json!({"a": 2, "b": 3}),
                expected_output: Some(serde_json::json!(5)),
            })
            .build();
        assert_eq!(tool.examples.len(), 1);
        assert_eq!(tool.examples[0].description, "Add 2 + 3");
    }

    #[test]
    fn test_tool_matches() {
        let tool = make_tool("calculator", "A math calculator", &["math", "utility"]);
        assert!(tool.matches("calc"));
        assert!(tool.matches("math"));
        assert!(tool.matches("utility"));
        assert!(!tool.matches("weather"));
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = ToolRegistry::new();
        reg.register(make_tool("add", "Add", &[]));
        reg.register(make_tool("sub", "Subtract", &[]));
        assert_eq!(reg.len(), 2);
        assert!(reg.get("add").is_some());
        assert!(reg.get("mul").is_none());
    }

    #[test]
    fn test_registry_search() {
        let mut reg = ToolRegistry::new();
        reg.register(make_tool("calculator", "Math calculator", &["math"]));
        reg.register(make_tool("weather", "Get weather", &["api"]));
        reg.register(make_tool("math_plot", "Plot math functions", &["math", "viz"]));
        let results = reg.search("math");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_registry_by_tag() {
        let mut reg = ToolRegistry::new();
        reg.register(make_tool("a", "Tool A", &["alpha", "beta"]));
        reg.register(make_tool("b", "Tool B", &["beta"]));
        reg.register(make_tool("c", "Tool C", &["gamma"]));
        assert_eq!(reg.by_tag("beta").len(), 2);
        assert_eq!(reg.by_tag("gamma").len(), 1);
        assert_eq!(reg.by_tag("delta").len(), 0);
    }

    #[test]
    fn test_registry_list() {
        let mut reg = ToolRegistry::new();
        reg.register(make_tool("x", "X", &[]));
        assert_eq!(reg.list().len(), 1);
        assert!(!reg.is_empty());
    }

    #[test]
    fn test_tool_example_serde() {
        let ex = ToolExample {
            description: "Test".to_string(),
            input: serde_json::json!({"x": 1}),
            expected_output: Some(serde_json::json!(2)),
        };
        let json = serde_json::to_string(&ex).unwrap();
        let back: ToolExample = serde_json::from_str(&json).unwrap();
        assert_eq!(back.description, "Test");
    }

    #[test]
    fn test_batch_tool_result_serde() {
        let r = BatchToolResult {
            tool_name: "add".to_string(),
            input: serde_json::json!({"a": 1}),
            output: Some(serde_json::json!(2)),
            error: None,
            duration_ms: 5,
        };
        let json = serde_json::to_string(&r).unwrap();
        let back: BatchToolResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tool_name, "add");
        assert_eq!(back.duration_ms, 5);
    }

    #[tokio::test]
    async fn test_batch_execute_all_success() {
        let mut reg = ToolRegistry::new();
        reg.register(
            Tool::builder("echo", "Echo input")
                .execute(|args| async move { Ok(args) })
                .build(),
        );
        let calls = vec![
            BatchToolCall {
                tool_name: "echo".to_string(),
                input: serde_json::json!({"msg": "a"}),
            },
            BatchToolCall {
                tool_name: "echo".to_string(),
                input: serde_json::json!({"msg": "b"}),
            },
        ];
        let results = batch_execute(&reg, calls, &BatchOptions::default()).await;
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.error.is_none()));
    }

    #[tokio::test]
    async fn test_batch_execute_tool_not_found() {
        let reg = ToolRegistry::new();
        let calls = vec![BatchToolCall {
            tool_name: "missing".to_string(),
            input: serde_json::json!({}),
        }];
        let results = batch_execute(&reg, calls, &BatchOptions::default()).await;
        assert_eq!(results.len(), 1);
        assert!(results[0].error.as_ref().unwrap().contains("not found"));
    }

    #[tokio::test]
    async fn test_batch_execute_with_concurrency() {
        let mut reg = ToolRegistry::new();
        reg.register(
            Tool::builder("slow", "Slow tool")
                .execute(|args| async move {
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                    Ok(args)
                })
                .build(),
        );
        let calls: Vec<_> = (0..5)
            .map(|i| BatchToolCall {
                tool_name: "slow".to_string(),
                input: serde_json::json!({"i": i}),
            })
            .collect();
        let opts = BatchOptions {
            concurrency: 2,
            continue_on_error: true,
        };
        let results = batch_execute(&reg, calls, &opts).await;
        assert_eq!(results.len(), 5);
        assert!(results.iter().all(|r| r.error.is_none()));
    }

    #[tokio::test]
    async fn test_batch_execute_with_error() {
        let mut reg = ToolRegistry::new();
        reg.register(
            Tool::builder("fail", "Always fails")
                .execute(|_args| async move {
                    Err(error::GaussError::tool("fail", "intentional error"))
                })
                .build(),
        );
        let calls = vec![BatchToolCall {
            tool_name: "fail".to_string(),
            input: serde_json::json!({}),
        }];
        let results = batch_execute(&reg, calls, &BatchOptions::default()).await;
        assert_eq!(results.len(), 1);
        assert!(results[0].error.is_some());
    }

    #[tokio::test]
    async fn test_batch_execute_mixed() {
        let mut reg = ToolRegistry::new();
        reg.register(
            Tool::builder("ok", "OK tool")
                .execute(|args| async move { Ok(args) })
                .build(),
        );
        let calls = vec![
            BatchToolCall {
                tool_name: "ok".to_string(),
                input: serde_json::json!({"v": 1}),
            },
            BatchToolCall {
                tool_name: "missing".to_string(),
                input: serde_json::json!({}),
            },
        ];
        let results = batch_execute(&reg, calls, &BatchOptions::default()).await;
        assert_eq!(results.len(), 2);
        assert!(results[0].error.is_none());
        assert!(results[1].error.is_some());
    }
}
