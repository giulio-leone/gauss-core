//! Provider resilience — fallback chains, circuit breaker, and composed resilience.
//!
//! Extends the existing `RetryProvider` with additional resilience patterns:
//! - `FallbackProvider`: try multiple providers in order
//! - `CircuitBreaker`: prevent cascade failures with open/half-open/closed states
//! - `ResilientProvider`: compose retry + fallback + circuit breaker

use async_trait::async_trait;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Duration;
use tracing::warn;

use crate::error::{self, GaussError};
use crate::message::Message;
use crate::provider::{BoxStream, GenerateOptions, GenerateResult, Provider};
use crate::tool::Tool;

// ---------------------------------------------------------------------------
// FallbackProvider
// ---------------------------------------------------------------------------

/// Tries multiple providers in order, falling back to the next on failure.
pub struct FallbackProvider {
    providers: Vec<crate::Shared<dyn Provider>>,
    fallback_on: FallbackPolicy,
}

/// Policy for when to fall back to the next provider.
#[derive(Debug, Clone, Default)]
pub enum FallbackPolicy {
    /// Fall back on any error.
    #[default]
    OnAnyError,
    /// Fall back only on specific error types.
    OnErrors(Vec<FallbackErrorKind>),
}

/// Error kinds that trigger a fallback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FallbackErrorKind {
    RateLimit,
    ServerError,
    Authentication,
    Timeout,
}

impl FallbackProvider {
    pub fn new(providers: Vec<crate::Shared<dyn Provider>>) -> Self {
        Self {
            providers,
            fallback_on: FallbackPolicy::default(),
        }
    }

    pub fn with_policy(mut self, policy: FallbackPolicy) -> Self {
        self.fallback_on = policy;
        self
    }

    fn should_fallback(&self, error: &GaussError) -> bool {
        match &self.fallback_on {
            FallbackPolicy::OnAnyError => true,
            FallbackPolicy::OnErrors(kinds) => {
                for kind in kinds {
                    let matches = match (kind, error) {
                        (FallbackErrorKind::RateLimit, GaussError::RateLimited { .. }) => true,
                        (FallbackErrorKind::ServerError, GaussError::Provider { status, .. }) => {
                            status.is_some_and(|s| s >= 500)
                        }
                        (FallbackErrorKind::Authentication, GaussError::Authentication { .. }) => {
                            true
                        }
                        (FallbackErrorKind::Timeout, GaussError::Timeout { .. }) => true,
                        _ => false,
                    };
                    if matches {
                        return true;
                    }
                }
                false
            }
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl Provider for FallbackProvider {
    fn name(&self) -> &str {
        "fallback"
    }

    fn model(&self) -> &str {
        self.providers
            .first()
            .map(|p| p.model())
            .unwrap_or("unknown")
    }

    async fn generate(
        &self,
        messages: &[Message],
        tools: &[Tool],
        options: &GenerateOptions,
    ) -> error::Result<GenerateResult> {
        let mut last_error = None;
        for (i, provider) in self.providers.iter().enumerate() {
            match provider.generate(messages, tools, options).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    if i + 1 < self.providers.len() && self.should_fallback(&e) {
                        warn!(
                            provider = provider.name(),
                            next = self.providers[i + 1].name(),
                            error = %e,
                            "Falling back to next provider"
                        );
                        last_error = Some(e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }
        Err(last_error
            .unwrap_or_else(|| GaussError::internal("No providers configured for fallback")))
    }

    async fn stream(
        &self,
        messages: &[Message],
        tools: &[Tool],
        options: &GenerateOptions,
    ) -> error::Result<BoxStream> {
        let mut last_error = None;
        for (i, provider) in self.providers.iter().enumerate() {
            match provider.stream(messages, tools, options).await {
                Ok(stream) => return Ok(stream),
                Err(e) => {
                    if i + 1 < self.providers.len() && self.should_fallback(&e) {
                        warn!(
                            provider = provider.name(),
                            next = self.providers[i + 1].name(),
                            error = %e,
                            "Falling back to next provider (stream)"
                        );
                        last_error = Some(e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }
        Err(last_error
            .unwrap_or_else(|| GaussError::internal("No providers configured for fallback")))
    }
}

// ---------------------------------------------------------------------------
// CircuitBreaker
// ---------------------------------------------------------------------------

/// Three-state circuit breaker: Closed (normal), Open (all calls fail fast),
/// HalfOpen (one probe call allowed).
pub struct CircuitBreaker {
    inner: crate::Shared<dyn Provider>,
    config: CircuitBreakerConfig,
    failure_count: AtomicU32,
    last_failure_time: AtomicU64,
    state: std::sync::atomic::AtomicU8,
}

/// Configuration for circuit breaker behavior.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures to trip the breaker.
    pub failure_threshold: u32,
    /// Time in ms to wait before allowing a probe in half-open state.
    pub recovery_timeout_ms: u64,
    /// Number of successful probes needed to close the breaker.
    pub success_threshold: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            recovery_timeout_ms: 30_000,
            success_threshold: 1,
        }
    }
}

/// Circuit breaker states.
const STATE_CLOSED: u8 = 0;
const STATE_OPEN: u8 = 1;
const STATE_HALF_OPEN: u8 = 2;

impl CircuitBreaker {
    pub fn new(inner: crate::Shared<dyn Provider>, config: CircuitBreakerConfig) -> Self {
        Self {
            inner,
            config,
            failure_count: AtomicU32::new(0),
            last_failure_time: AtomicU64::new(0),
            state: std::sync::atomic::AtomicU8::new(STATE_CLOSED),
        }
    }

    pub fn wrap(inner: crate::Shared<dyn Provider>) -> Self {
        Self::new(inner, CircuitBreakerConfig::default())
    }

    fn now_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64
    }

    fn current_state(&self) -> u8 {
        let state = self.state.load(Ordering::Relaxed);
        if state == STATE_OPEN {
            let elapsed = Self::now_ms() - self.last_failure_time.load(Ordering::Relaxed);
            if elapsed >= self.config.recovery_timeout_ms {
                self.state.store(STATE_HALF_OPEN, Ordering::Relaxed);
                return STATE_HALF_OPEN;
            }
        }
        state
    }

    fn record_success(&self) {
        self.failure_count.store(0, Ordering::Relaxed);
        self.state.store(STATE_CLOSED, Ordering::Relaxed);
    }

    fn record_failure(&self) {
        let failures = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
        self.last_failure_time
            .store(Self::now_ms(), Ordering::Relaxed);
        if failures >= self.config.failure_threshold {
            self.state.store(STATE_OPEN, Ordering::Relaxed);
        }
    }

    /// Get the current state as a human-readable string.
    pub fn state_name(&self) -> &'static str {
        match self.current_state() {
            STATE_CLOSED => "closed",
            STATE_OPEN => "open",
            STATE_HALF_OPEN => "half_open",
            _ => "unknown",
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl Provider for CircuitBreaker {
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
        let state = self.current_state();
        if state == STATE_OPEN {
            return Err(GaussError::internal(format!(
                "Circuit breaker is open for provider '{}'",
                self.inner.name()
            )));
        }

        match self.inner.generate(messages, tools, options).await {
            Ok(result) => {
                self.record_success();
                Ok(result)
            }
            Err(e) => {
                self.record_failure();
                Err(e)
            }
        }
    }

    async fn stream(
        &self,
        messages: &[Message],
        tools: &[Tool],
        options: &GenerateOptions,
    ) -> error::Result<BoxStream> {
        let state = self.current_state();
        if state == STATE_OPEN {
            return Err(GaussError::internal(format!(
                "Circuit breaker is open for provider '{}'",
                self.inner.name()
            )));
        }

        match self.inner.stream(messages, tools, options).await {
            Ok(stream) => {
                self.record_success();
                Ok(stream)
            }
            Err(e) => {
                self.record_failure();
                Err(e)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ResilientProvider
// ---------------------------------------------------------------------------

/// Builder for composing resilience strategies (retry + fallback + circuit breaker).
pub struct ResilientProviderBuilder {
    primary: crate::Shared<dyn Provider>,
    fallbacks: Vec<crate::Shared<dyn Provider>>,
    retry_config: Option<crate::provider::retry::RetryConfig>,
    circuit_breaker_config: Option<CircuitBreakerConfig>,
    fallback_policy: FallbackPolicy,
}

impl ResilientProviderBuilder {
    pub fn new(primary: crate::Shared<dyn Provider>) -> Self {
        Self {
            primary,
            fallbacks: Vec::new(),
            retry_config: None,
            circuit_breaker_config: None,
            fallback_policy: FallbackPolicy::default(),
        }
    }

    /// Add a fallback provider.
    pub fn fallback(mut self, provider: crate::Shared<dyn Provider>) -> Self {
        self.fallbacks.push(provider);
        self
    }

    /// Set retry configuration.
    pub fn retry(mut self, config: crate::provider::retry::RetryConfig) -> Self {
        self.retry_config = Some(config);
        self
    }

    /// Set circuit breaker configuration.
    pub fn circuit_breaker(mut self, config: CircuitBreakerConfig) -> Self {
        self.circuit_breaker_config = Some(config);
        self
    }

    /// Set fallback policy.
    pub fn fallback_policy(mut self, policy: FallbackPolicy) -> Self {
        self.fallback_policy = policy;
        self
    }

    /// Build the composed resilient provider.
    /// Wrapping order: CircuitBreaker → Retry → Fallback (innermost to outermost).
    pub fn build(self) -> crate::Shared<dyn Provider> {
        let mut provider: crate::Shared<dyn Provider> = self.primary;

        // Wrap with circuit breaker (innermost)
        if let Some(cb_config) = self.circuit_breaker_config {
            provider = crate::Shared::new(CircuitBreaker::new(provider, cb_config));
        }

        // Wrap with retry
        if let Some(retry_config) = self.retry_config {
            provider = crate::Shared::new(crate::provider::retry::RetryProvider::new(
                provider,
                retry_config,
            ));
        }

        // Wrap with fallback (outermost)
        if !self.fallbacks.is_empty() {
            let mut all = vec![provider];
            all.extend(self.fallbacks);
            provider =
                crate::Shared::new(FallbackProvider::new(all).with_policy(self.fallback_policy));
        }

        provider
    }
}

/// Convenience function to create a resilient provider builder.
pub fn resilient(primary: crate::Shared<dyn Provider>) -> ResilientProviderBuilder {
    ResilientProviderBuilder::new(primary)
}
