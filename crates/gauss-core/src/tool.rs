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
    dyn Fn(
            serde_json::Value,
        ) -> Pin<Box<dyn Future<Output = error::Result<serde_json::Value>>>>
>;

/// A tool that can be used by an agent.
#[derive(Clone)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: ToolParameters,
    execute: Option<ToolExecuteFn>,
}

impl std::fmt::Debug for Tool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("parameters", &self.parameters)
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
}

pub struct ToolBuilder {
    name: String,
    description: String,
    parameters: ToolParameters,
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
            execute: self.execute,
        }
    }
}
