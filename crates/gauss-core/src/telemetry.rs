//! Telemetry & observability — span-based tracing, metrics collection.
//!
//! Uses the `tracing` crate ecosystem for structured logging and spans.
//! Can bridge to OpenTelemetry via `tracing-opentelemetry`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Span & Event types
// ---------------------------------------------------------------------------

/// A recorded span representing a unit of work.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanRecord {
    pub name: String,
    pub span_type: SpanType,
    pub start_ms: u64,
    pub duration_ms: u64,
    pub attributes: HashMap<String, serde_json::Value>,
    pub status: SpanStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub children: Vec<SpanRecord>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpanType {
    AgentRun,
    ModelCall,
    ToolCall,
    WorkflowStep,
    MiddlewareHook,
    Embedding,
    VectorSearch,
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpanStatus {
    Ok,
    Error,
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Collected metrics for an agent run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentMetrics {
    pub total_steps: usize,
    pub total_tool_calls: usize,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_reasoning_tokens: u64,
    pub total_cache_tokens: u64,
    pub total_duration_ms: u64,
    pub model_call_count: usize,
    pub model_call_latencies_ms: Vec<u64>,
    pub tool_call_latencies_ms: Vec<u64>,
    pub errors: Vec<String>,
}

impl AgentMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Average model call latency in ms.
    pub fn avg_model_latency_ms(&self) -> f64 {
        if self.model_call_latencies_ms.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.model_call_latencies_ms.iter().sum();
        sum as f64 / self.model_call_latencies_ms.len() as f64
    }

    /// Average tool call latency in ms.
    pub fn avg_tool_latency_ms(&self) -> f64 {
        if self.tool_call_latencies_ms.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.tool_call_latencies_ms.iter().sum();
        sum as f64 / self.tool_call_latencies_ms.len() as f64
    }

    /// Record a model call.
    pub fn record_model_call(&mut self, duration_ms: u64, input_tokens: u64, output_tokens: u64) {
        self.model_call_count += 1;
        self.model_call_latencies_ms.push(duration_ms);
        self.total_input_tokens += input_tokens;
        self.total_output_tokens += output_tokens;
    }

    /// Record a tool call.
    pub fn record_tool_call(&mut self, duration_ms: u64) {
        self.total_tool_calls += 1;
        self.tool_call_latencies_ms.push(duration_ms);
    }

    pub fn record_error(&mut self, error: String) {
        self.errors.push(error);
    }
}

// ---------------------------------------------------------------------------
// Span Builder (for easy instrumentation)
// ---------------------------------------------------------------------------

/// Builder for creating timed spans.
#[derive(Debug)]
pub struct SpanBuilder {
    name: String,
    span_type: SpanType,
    start: Instant,
    attributes: HashMap<String, serde_json::Value>,
    children: Vec<SpanRecord>,
}

impl SpanBuilder {
    pub fn new(name: impl Into<String>, span_type: SpanType) -> Self {
        Self {
            name: name.into(),
            span_type,
            start: Instant::now(),
            attributes: HashMap::new(),
            children: Vec::new(),
        }
    }

    pub fn attribute(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    pub fn child(mut self, record: SpanRecord) -> Self {
        self.children.push(record);
        self
    }

    /// Finish the span successfully.
    pub fn finish(self) -> SpanRecord {
        SpanRecord {
            name: self.name,
            span_type: self.span_type,
            start_ms: 0, // relative to run start
            duration_ms: self.start.elapsed().as_millis() as u64,
            attributes: self.attributes,
            status: SpanStatus::Ok,
            error: None,
            children: self.children,
        }
    }

    /// Finish the span with an error.
    pub fn finish_with_error(self, error: impl Into<String>) -> SpanRecord {
        SpanRecord {
            name: self.name,
            span_type: self.span_type,
            start_ms: 0,
            duration_ms: self.start.elapsed().as_millis() as u64,
            attributes: self.attributes,
            status: SpanStatus::Error,
            error: Some(error.into()),
            children: self.children,
        }
    }
}

// ---------------------------------------------------------------------------
// Telemetry Collector
// ---------------------------------------------------------------------------

/// Collects spans and metrics for an agent session.
#[derive(Debug, Default)]
pub struct TelemetryCollector {
    spans: std::sync::Mutex<Vec<SpanRecord>>,
    metrics: std::sync::Mutex<AgentMetrics>,
}

impl TelemetryCollector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a completed span.
    pub fn record_span(&self, span: SpanRecord) {
        if let Ok(mut spans) = self.spans.lock() {
            spans.push(span);
        }
    }

    /// Get a mutable reference to metrics for recording.
    pub fn with_metrics<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut AgentMetrics) -> R,
    {
        let mut metrics = self.metrics.lock().expect("metrics lock");
        f(&mut metrics)
    }

    /// Export all collected spans.
    pub fn export_spans(&self) -> Vec<SpanRecord> {
        self.spans.lock().map(|s| s.clone()).unwrap_or_default()
    }

    /// Export current metrics snapshot.
    pub fn export_metrics(&self) -> AgentMetrics {
        self.metrics.lock().map(|m| m.clone()).unwrap_or_default()
    }

    /// Clear all collected data.
    pub fn clear(&self) {
        if let Ok(mut spans) = self.spans.lock() {
            spans.clear();
        }
        if let Ok(mut metrics) = self.metrics.lock() {
            *metrics = AgentMetrics::default();
        }
    }
}
