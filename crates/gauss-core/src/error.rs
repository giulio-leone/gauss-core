use thiserror::Error;

/// Root error type for all Gauss operations.
#[derive(Error, Debug)]
pub enum GaussError {
    #[error("Provider error: {message}")]
    Provider {
        message: String,
        status: Option<u16>,
        provider: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Agent error: {message}")]
    Agent {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Tool error in '{tool_name}': {message}")]
    Tool {
        tool_name: String,
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Stream error: {message}")]
    Stream {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Configuration error: {message}")]
    Config { message: String },

    #[error("Schema validation error: {message}")]
    SchemaValidation { message: String },

    #[error("Rate limited by provider '{provider}', retry after {retry_after_ms:?}ms")]
    RateLimited {
        provider: String,
        retry_after_ms: Option<u64>,
    },

    #[error("Authentication failed for provider '{provider}'")]
    Authentication { provider: String },

    #[error("Request aborted")]
    Aborted,

    #[error("Timeout after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    #[error("No content generated")]
    NoContentGenerated,

    #[error("Guardrail blocked: {reason}")]
    Guardrail {
        reason: String,
        guardrail: Option<String>,
    },

    #[error("Circuit breaker open for provider '{provider}'")]
    CircuitBreakerOpen { provider: String },

    #[error("Plugin error in '{plugin}': {message}")]
    PluginError { plugin: String, message: String },

    #[error("Internal error: {message}")]
    Internal { message: String },
}

impl GaussError {
    pub fn provider(provider: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Provider {
            message: message.into(),
            status: None,
            provider: provider.into(),
            source: None,
        }
    }

    pub fn agent(message: impl Into<String>) -> Self {
        Self::Agent {
            message: message.into(),
            source: None,
        }
    }

    pub fn tool(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Tool {
            tool_name: name.into(),
            message: message.into(),
            source: None,
        }
    }

    pub fn rate_limited(provider: impl Into<String>, _message: impl Into<String>) -> Self {
        Self::RateLimited {
            provider: provider.into(),
            retry_after_ms: None,
        }
    }

    pub fn authentication(provider: impl Into<String>, _message: impl Into<String>) -> Self {
        Self::Authentication {
            provider: provider.into(),
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }
}

pub type Result<T> = std::result::Result<T, GaussError>;
