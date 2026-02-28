use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, warn};

use crate::error::{self, GaussError};
use crate::message::Message;
use crate::provider::{GenerateOptions, GenerateResult, Provider};
use crate::streaming::StreamEvent;
use crate::tool::Tool;

/// Configuration for retry behavior.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    pub backoff_multiplier: f64,
    pub retry_on_rate_limit: bool,
    pub retry_on_server_error: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay_ms: 1000,
            max_delay_ms: 60_000,
            backoff_multiplier: 2.0,
            retry_on_rate_limit: true,
            retry_on_server_error: true,
        }
    }
}

/// A provider wrapper that adds retry logic with exponential backoff.
pub struct RetryProvider {
    inner: Arc<dyn Provider>,
    config: RetryConfig,
}

impl RetryProvider {
    pub fn new(inner: Arc<dyn Provider>, config: RetryConfig) -> Self {
        Self { inner, config }
    }

    pub fn wrap(inner: Arc<dyn Provider>) -> Self {
        Self::new(inner, RetryConfig::default())
    }

    fn should_retry(&self, error: &GaussError) -> bool {
        match error {
            GaussError::RateLimited { .. } => self.config.retry_on_rate_limit,
            GaussError::Provider { status, .. } => {
                self.config.retry_on_server_error && status.is_some_and(|s| s >= 500)
            }
            GaussError::Stream { .. } => true,
            _ => false,
        }
    }

    fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let delay = self.config.initial_delay_ms as f64
            * self.config.backoff_multiplier.powi(attempt as i32);
        let clamped = delay.min(self.config.max_delay_ms as f64) as u64;
        Duration::from_millis(clamped)
    }
}

#[async_trait]
impl Provider for RetryProvider {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn model(&self) -> &str {
        self.inner.model()
    }

    async fn generate(
        &self,
        messages: &[Message],
        tools: &[Tool],
        options: &GenerateOptions,
    ) -> error::Result<GenerateResult> {
        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            match self.inner.generate(messages, tools, options).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    if attempt < self.config.max_retries && self.should_retry(&e) {
                        let delay = self.delay_for_attempt(attempt);
                        warn!(
                            provider = self.inner.name(),
                            attempt = attempt + 1,
                            max = self.config.max_retries,
                            delay_ms = delay.as_millis() as u64,
                            error = %e,
                            "Retrying after error"
                        );
                        tokio::time::sleep(delay).await;
                        last_error = Some(e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Err(last_error
            .unwrap_or_else(|| GaussError::provider(self.inner.name(), "Max retries exceeded")))
    }

    async fn stream(
        &self,
        messages: &[Message],
        tools: &[Tool],
        options: &GenerateOptions,
    ) -> error::Result<Box<dyn futures::Stream<Item = error::Result<StreamEvent>> + Send + Unpin>>
    {
        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            match self.inner.stream(messages, tools, options).await {
                Ok(stream) => {
                    debug!(provider = self.inner.name(), "Stream established");
                    return Ok(stream);
                }
                Err(e) => {
                    if attempt < self.config.max_retries && self.should_retry(&e) {
                        let delay = self.delay_for_attempt(attempt);
                        warn!(
                            provider = self.inner.name(),
                            attempt = attempt + 1,
                            delay_ms = delay.as_millis() as u64,
                            "Retrying stream after error"
                        );
                        tokio::time::sleep(delay).await;
                        last_error = Some(e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Err(last_error
            .unwrap_or_else(|| GaussError::provider(self.inner.name(), "Max retries exceeded")))
    }
}
