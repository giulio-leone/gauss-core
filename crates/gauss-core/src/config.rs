//! Declarative agent configuration from YAML/TOML/JSON.
//!
//! Enables Agent::from_config() to create agents from config files,
//! supporting the full agent builder API declaratively.

use serde::{Deserialize, Serialize};

use crate::error::{self, GaussError};
use crate::provider::ProviderConfig;
use crate::tool::{ToolChoice, ToolParameters};

/// Top-level agent configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub name: String,
    pub provider: ProviderConfigDef,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(default)]
    pub tools: Vec<ToolConfigDef>,
    #[serde(default = "default_max_steps")]
    pub max_steps: usize,
    #[serde(default)]
    pub options: AgentOptionsDef,
    #[serde(default)]
    pub stop_conditions: Vec<StopConditionDef>,
}

fn default_max_steps() -> usize {
    10
}

/// Provider definition in config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfigDef {
    /// Provider type: openai, anthropic, google, groq, ollama, deepseek
    #[serde(rename = "type")]
    pub provider_type: String,
    pub model: String,
    /// API key (can use env var syntax: ${OPENAI_API_KEY})
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_retries: Option<u32>,
}

/// Tool definition in config (declarative, no execute fn).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolConfigDef {
    pub name: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// Generation options.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentOptionsDef {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoiceDef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<serde_json::Value>,
}

/// Tool choice as a string in config.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceDef {
    Auto,
    None,
    Required,
    Specific(String),
}

impl From<ToolChoiceDef> for ToolChoice {
    fn from(def: ToolChoiceDef) -> Self {
        match def {
            ToolChoiceDef::Auto => ToolChoice::Auto,
            ToolChoiceDef::None => ToolChoice::None,
            ToolChoiceDef::Required => ToolChoice::Required,
            ToolChoiceDef::Specific(name) => ToolChoice::Specific { name },
        }
    }
}

/// Stop condition in config.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopConditionDef {
    MaxSteps(usize),
    HasToolCall(String),
    TextGenerated,
    Custom(String),
}

impl From<StopConditionDef> for crate::agent::StopCondition {
    fn from(def: StopConditionDef) -> Self {
        match def {
            StopConditionDef::MaxSteps(n) => crate::agent::StopCondition::MaxSteps(n),
            StopConditionDef::HasToolCall(name) => crate::agent::StopCondition::HasToolCall(name),
            StopConditionDef::TextGenerated => crate::agent::StopCondition::TextGenerated,
            StopConditionDef::Custom(name) => crate::agent::StopCondition::Custom(name),
        }
    }
}

/// Resolve environment variable references like ${VAR_NAME}.
pub fn resolve_env(value: &str) -> String {
    if let Some(stripped) = value.strip_prefix("${")
        && let Some(var_name) = stripped.strip_suffix('}')
    {
        return std::env::var(var_name).unwrap_or_default();
    }
    value.to_string()
}

impl AgentConfig {
    /// Parse from YAML string.
    #[cfg(feature = "config-yaml")]
    pub fn from_yaml(yaml: &str) -> error::Result<Self> {
        serde_yaml::from_str(yaml).map_err(|e| GaussError::Config {
            message: format!("Invalid YAML config: {e}"),
        })
    }

    /// Parse from TOML string.
    #[cfg(feature = "config-toml")]
    pub fn from_toml(toml: &str) -> error::Result<Self> {
        toml::from_str(toml).map_err(|e| GaussError::Config {
            message: format!("Invalid TOML config: {e}"),
        })
    }

    /// Parse from JSON string.
    pub fn from_json(json: &str) -> error::Result<Self> {
        serde_json::from_str(json).map_err(|e| GaussError::Config {
            message: format!("Invalid JSON config: {e}"),
        })
    }

    /// Detect format from file extension and parse.
    pub fn from_file(path: &str) -> error::Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| GaussError::Config {
            message: format!("Failed to read config file '{path}': {e}"),
        })?;

        if path.ends_with(".json") {
            Self::from_json(&content)
        } else {
            #[cfg(feature = "config-yaml")]
            if path.ends_with(".yaml") || path.ends_with(".yml") {
                return Self::from_yaml(&content);
            }
            #[cfg(feature = "config-toml")]
            if path.ends_with(".toml") {
                return Self::from_toml(&content);
            }
            // Default: try JSON
            Self::from_json(&content)
        }
    }

    /// Resolve the provider config (including env var substitution).
    pub fn resolve_provider_config(&self) -> ProviderConfig {
        let api_key = self
            .provider
            .api_key
            .as_deref()
            .map(resolve_env)
            .unwrap_or_default();

        let mut config = ProviderConfig::new(api_key);
        if let Some(ref url) = self.provider.base_url {
            config.base_url = Some(resolve_env(url));
        }
        if let Some(timeout) = self.provider.timeout_ms {
            config.timeout_ms = Some(timeout);
        }
        if let Some(retries) = self.provider.max_retries {
            config.max_retries = Some(retries);
        }
        config
    }

    /// Build tool definitions (without execute fns — those must be registered separately).
    pub fn build_tools(&self) -> Vec<crate::tool::Tool> {
        self.tools
            .iter()
            .map(|td| {
                let mut builder = crate::tool::Tool::builder(&td.name, &td.description);
                if let Some(ref params) = td.parameters
                    && let Ok(p) = serde_json::from_value::<ToolParameters>(params.clone())
                {
                    builder = builder.parameters(p);
                }
                builder.build()
            })
            .collect()
    }

    /// Build generation options from config.
    pub fn build_options(&self) -> crate::provider::GenerateOptions {
        crate::provider::GenerateOptions {
            temperature: self.options.temperature,
            top_p: self.options.top_p,
            top_k: self.options.top_k,
            max_tokens: self.options.max_tokens,
            seed: self.options.seed,
            frequency_penalty: self.options.frequency_penalty,
            presence_penalty: self.options.presence_penalty,
            output_schema: self.options.output_schema.clone(),
            tool_choice: self
                .options
                .tool_choice
                .as_ref()
                .map(|tc| tc.clone().into()),
            ..Default::default()
        }
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> error::Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| GaussError::Config {
            message: format!("Failed to serialize config: {e}"),
        })
    }
}
