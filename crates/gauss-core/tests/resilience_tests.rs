use async_trait::async_trait;
use gauss_core::error::{GaussError, Result};
use gauss_core::message::{Message, Usage};
use gauss_core::provider::{BoxStream, FinishReason, GenerateOptions, GenerateResult, Provider};
use gauss_core::resilience::*;
use gauss_core::tool::Tool;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

/// Mock provider that fails N times then succeeds.
struct FailThenSucceed {
    name: String,
    failures_left: AtomicU32,
}

impl FailThenSucceed {
    fn new(name: &str, fail_count: u32) -> Self {
        Self {
            name: name.to_string(),
            failures_left: AtomicU32::new(fail_count),
        }
    }
}

#[async_trait]
impl Provider for FailThenSucceed {
    fn name(&self) -> &str {
        &self.name
    }

    fn model(&self) -> &str {
        "mock"
    }

    async fn generate(
        &self,
        _messages: &[Message],
        _tools: &[Tool],
        _options: &GenerateOptions,
    ) -> Result<GenerateResult> {
        let left = self.failures_left.fetch_sub(1, Ordering::Relaxed);
        if left > 0 {
            Err(GaussError::provider(&self.name, "Simulated failure"))
        } else {
            Ok(GenerateResult {
                message: Message::assistant(format!("Success from {}", self.name)),
                usage: Usage::default(),
                finish_reason: FinishReason::Stop,
                provider_metadata: serde_json::json!({}),
                thinking: None, citations: vec![], grounding_metadata: None,
            })
        }
    }

    async fn stream(
        &self,
        _messages: &[Message],
        _tools: &[Tool],
        _options: &GenerateOptions,
    ) -> Result<BoxStream> {
        Err(GaussError::provider(&self.name, "Stream not supported"))
    }
}

/// Mock provider that always fails.
struct AlwaysFails {
    name: String,
}

impl AlwaysFails {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }
}

#[async_trait]
impl Provider for AlwaysFails {
    fn name(&self) -> &str {
        &self.name
    }

    fn model(&self) -> &str {
        "mock"
    }

    async fn generate(
        &self,
        _messages: &[Message],
        _tools: &[Tool],
        _options: &GenerateOptions,
    ) -> Result<GenerateResult> {
        Err(GaussError::provider(&self.name, "Always fails"))
    }

    async fn stream(
        &self,
        _messages: &[Message],
        _tools: &[Tool],
        _options: &GenerateOptions,
    ) -> Result<BoxStream> {
        Err(GaussError::provider(&self.name, "Always fails"))
    }
}

/// Mock provider that always succeeds.
struct AlwaysSucceeds {
    name: String,
}

impl AlwaysSucceeds {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }
}

#[async_trait]
impl Provider for AlwaysSucceeds {
    fn name(&self) -> &str {
        &self.name
    }

    fn model(&self) -> &str {
        "mock"
    }

    async fn generate(
        &self,
        _messages: &[Message],
        _tools: &[Tool],
        _options: &GenerateOptions,
    ) -> Result<GenerateResult> {
        Ok(GenerateResult {
            message: Message::assistant(format!("Success from {}", self.name)),
            usage: Usage::default(),
            finish_reason: FinishReason::Stop,
            provider_metadata: serde_json::json!({}),
            thinking: None, citations: vec![], grounding_metadata: None,
        })
    }

    async fn stream(
        &self,
        _messages: &[Message],
        _tools: &[Tool],
        _options: &GenerateOptions,
    ) -> Result<BoxStream> {
        Err(GaussError::provider(&self.name, "Not implemented"))
    }
}

#[tokio::test]
async fn fallback_uses_second_provider() {
    let primary: Arc<dyn Provider> = Arc::new(AlwaysFails::new("primary"));
    let backup: Arc<dyn Provider> = Arc::new(AlwaysSucceeds::new("backup"));

    let fallback = FallbackProvider::new(vec![primary, backup]);
    let result = fallback
        .generate(&[Message::user("hi")], &[], &GenerateOptions::default())
        .await
        .unwrap();

    assert_eq!(result.message.text().unwrap(), "Success from backup");
}

#[tokio::test]
async fn fallback_uses_first_if_succeeds() {
    let primary: Arc<dyn Provider> = Arc::new(AlwaysSucceeds::new("primary"));
    let backup: Arc<dyn Provider> = Arc::new(AlwaysSucceeds::new("backup"));

    let fallback = FallbackProvider::new(vec![primary, backup]);
    let result = fallback
        .generate(&[Message::user("hi")], &[], &GenerateOptions::default())
        .await
        .unwrap();

    assert_eq!(result.message.text().unwrap(), "Success from primary");
}

#[tokio::test]
async fn fallback_all_fail() {
    let p1: Arc<dyn Provider> = Arc::new(AlwaysFails::new("p1"));
    let p2: Arc<dyn Provider> = Arc::new(AlwaysFails::new("p2"));

    let fallback = FallbackProvider::new(vec![p1, p2]);
    let err = fallback
        .generate(&[Message::user("hi")], &[], &GenerateOptions::default())
        .await
        .unwrap_err();

    assert!(format!("{err}").contains("Always fails"));
}

#[tokio::test]
async fn circuit_breaker_opens_after_threshold() {
    let inner: Arc<dyn Provider> = Arc::new(AlwaysFails::new("failing"));
    let cb = CircuitBreaker::new(
        inner,
        CircuitBreakerConfig {
            failure_threshold: 3,
            recovery_timeout_ms: 60_000,
            success_threshold: 1,
        },
    );

    // First 3 calls fail normally
    for _ in 0..3 {
        let _ = cb
            .generate(&[Message::user("hi")], &[], &GenerateOptions::default())
            .await;
    }

    assert_eq!(cb.state_name(), "open");

    // Next call should fail fast
    let err = cb
        .generate(&[Message::user("hi")], &[], &GenerateOptions::default())
        .await
        .unwrap_err();
    assert!(format!("{err}").contains("Circuit breaker is open"));
}

#[tokio::test]
async fn circuit_breaker_resets_on_success() {
    let inner: Arc<dyn Provider> = Arc::new(FailThenSucceed::new("flaky", 2));
    let cb = CircuitBreaker::new(
        inner,
        CircuitBreakerConfig {
            failure_threshold: 5,
            recovery_timeout_ms: 60_000,
            success_threshold: 1,
        },
    );

    // Fail twice
    let _ = cb
        .generate(&[Message::user("hi")], &[], &GenerateOptions::default())
        .await;
    let _ = cb
        .generate(&[Message::user("hi")], &[], &GenerateOptions::default())
        .await;

    // Should still be closed (threshold is 5)
    assert_eq!(cb.state_name(), "closed");

    // Third call succeeds, resets counter
    let result = cb
        .generate(&[Message::user("hi")], &[], &GenerateOptions::default())
        .await
        .unwrap();
    assert!(result.message.text().unwrap().contains("Success"));
    assert_eq!(cb.state_name(), "closed");
}

#[tokio::test]
async fn resilient_builder() {
    let primary: Arc<dyn Provider> = Arc::new(AlwaysFails::new("primary"));
    let backup: Arc<dyn Provider> = Arc::new(AlwaysSucceeds::new("backup"));

    let provider = resilient(primary).fallback(backup).build();

    let result = provider
        .generate(&[Message::user("hi")], &[], &GenerateOptions::default())
        .await
        .unwrap();

    assert_eq!(result.message.text().unwrap(), "Success from backup");
}

#[tokio::test]
async fn resilient_with_circuit_breaker_and_fallback() {
    let primary: Arc<dyn Provider> = Arc::new(AlwaysFails::new("primary"));
    let backup: Arc<dyn Provider> = Arc::new(AlwaysSucceeds::new("backup"));

    let provider = resilient(primary)
        .circuit_breaker(CircuitBreakerConfig {
            failure_threshold: 2,
            recovery_timeout_ms: 60_000,
            success_threshold: 1,
        })
        .fallback(backup)
        .build();

    // Should fall back to backup
    let result = provider
        .generate(&[Message::user("hi")], &[], &GenerateOptions::default())
        .await
        .unwrap();

    assert_eq!(result.message.text().unwrap(), "Success from backup");
}
