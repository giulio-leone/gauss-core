//! Stream transformers — composable pipeline for transforming agent stream events.
//!
//! Provides partial JSON parsing for structured output streaming and
//! a transformer pipeline (map, filter, tap, accumulate).

use crate::streaming::StreamEvent;

// ---------------------------------------------------------------------------
// Partial JSON Parser
// ---------------------------------------------------------------------------

/// Incrementally parse partial JSON from accumulated text.
/// Returns `Some(Value)` for the largest valid partial parse, or `None`.
pub fn parse_partial_json(text: &str) -> Option<serde_json::Value> {
    // Try full parse first
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(text) {
        return Some(v);
    }

    // Try closing open braces/brackets
    let mut attempt = text.to_string();
    let mut open_braces = 0i32;
    let mut open_brackets = 0i32;
    let mut in_string = false;
    let mut escape = false;

    for ch in text.chars() {
        if escape {
            escape = false;
            continue;
        }
        match ch {
            '\\' if in_string => escape = true,
            '"' => in_string = !in_string,
            '{' if !in_string => open_braces += 1,
            '}' if !in_string => open_braces -= 1,
            '[' if !in_string => open_brackets += 1,
            ']' if !in_string => open_brackets -= 1,
            _ => {}
        }
    }

    // If we're inside a string, close it
    if in_string {
        attempt.push('"');
    }

    // Remove trailing comma before closing
    let trimmed = attempt.trim_end();
    if let Some(stripped) = trimmed.strip_suffix(',') {
        attempt = stripped.to_string();
    }

    // Close open brackets/braces
    for _ in 0..open_brackets {
        attempt.push(']');
    }
    for _ in 0..open_braces {
        attempt.push('}');
    }

    serde_json::from_str::<serde_json::Value>(&attempt).ok()
}

// ---------------------------------------------------------------------------
// ObjectAccumulator
// ---------------------------------------------------------------------------

/// Accumulates text deltas and emits partial JSON objects as they become parseable.
#[derive(Debug, Default)]
pub struct ObjectAccumulator {
    buffer: String,
    last_emitted: Option<serde_json::Value>,
}

impl ObjectAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a text delta. Returns `Some(partial_value)` if a new partial parse succeeded
    /// and differs from the last emitted value.
    pub fn feed(&mut self, delta: &str) -> Option<serde_json::Value> {
        self.buffer.push_str(delta);

        let partial = parse_partial_json(&self.buffer)?;

        // Only emit if different from last
        if self.last_emitted.as_ref() == Some(&partial) {
            return None;
        }

        self.last_emitted = Some(partial.clone());
        Some(partial)
    }

    /// Get the final accumulated text.
    pub fn text(&self) -> &str {
        &self.buffer
    }

    /// Get the last successfully parsed partial value.
    pub fn last_value(&self) -> Option<&serde_json::Value> {
        self.last_emitted.as_ref()
    }

    /// Reset the accumulator.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.last_emitted = None;
    }
}

// ---------------------------------------------------------------------------
// StreamTransformer Trait
// ---------------------------------------------------------------------------

/// Transforms stream events. Return `None` to filter out an event.
pub trait StreamTransformer: Send + Sync {
    fn transform(&mut self, event: StreamEvent) -> Option<StreamEvent>;
}

// ---------------------------------------------------------------------------
// Built-in Transformers
// ---------------------------------------------------------------------------

/// Maps text deltas through a function.
pub struct MapText<F: Fn(&str) -> String + Send + Sync> {
    f: F,
}

impl<F: Fn(&str) -> String + Send + Sync> MapText<F> {
    pub fn new(f: F) -> Self {
        Self { f }
    }
}

impl<F: Fn(&str) -> String + Send + Sync> StreamTransformer for MapText<F> {
    fn transform(&mut self, event: StreamEvent) -> Option<StreamEvent> {
        match event {
            StreamEvent::TextDelta(text) => Some(StreamEvent::TextDelta((self.f)(&text))),
            other => Some(other),
        }
    }
}

/// Filters events by predicate.
pub struct FilterEvents<F: Fn(&StreamEvent) -> bool + Send + Sync> {
    predicate: F,
}

impl<F: Fn(&StreamEvent) -> bool + Send + Sync> FilterEvents<F> {
    pub fn new(predicate: F) -> Self {
        Self { predicate }
    }
}

impl<F: Fn(&StreamEvent) -> bool + Send + Sync> StreamTransformer for FilterEvents<F> {
    fn transform(&mut self, event: StreamEvent) -> Option<StreamEvent> {
        if (self.predicate)(&event) {
            Some(event)
        } else {
            None
        }
    }
}

/// Side-effect transformer — observes events without modifying them.
pub struct Tap<F: FnMut(&StreamEvent) + Send + Sync> {
    f: F,
}

impl<F: FnMut(&StreamEvent) + Send + Sync> Tap<F> {
    pub fn new(f: F) -> Self {
        Self { f }
    }
}

impl<F: FnMut(&StreamEvent) + Send + Sync> StreamTransformer for Tap<F> {
    fn transform(&mut self, event: StreamEvent) -> Option<StreamEvent> {
        (self.f)(&event);
        Some(event)
    }
}

/// Accumulates text deltas and emits ObjectDelta events for partial JSON.
pub struct ObjectDeltaTransformer {
    accumulator: ObjectAccumulator,
}

impl ObjectDeltaTransformer {
    pub fn new() -> Self {
        Self {
            accumulator: ObjectAccumulator::new(),
        }
    }
}

impl Default for ObjectDeltaTransformer {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamTransformer for ObjectDeltaTransformer {
    fn transform(&mut self, event: StreamEvent) -> Option<StreamEvent> {
        match &event {
            StreamEvent::TextDelta(delta) => {
                if let Some(partial) = self.accumulator.feed(delta) {
                    Some(StreamEvent::ObjectDelta(partial))
                } else {
                    Some(event)
                }
            }
            _ => Some(event),
        }
    }
}

// ---------------------------------------------------------------------------
// StreamPipeline
// ---------------------------------------------------------------------------

/// Composes multiple transformers into a pipeline.
pub struct StreamPipeline {
    transformers: Vec<Box<dyn StreamTransformer>>,
}

impl Default for StreamPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamPipeline {
    pub fn new() -> Self {
        Self {
            transformers: Vec::new(),
        }
    }

    /// Add a transformer to the pipeline.
    pub fn pipe(mut self, transformer: Box<dyn StreamTransformer>) -> Self {
        self.transformers.push(transformer);
        self
    }

    /// Transform an event through the entire pipeline.
    pub fn transform(&mut self, event: StreamEvent) -> Option<StreamEvent> {
        let mut current = Some(event);
        for t in &mut self.transformers {
            current = current.and_then(|e| t.transform(e));
        }
        current
    }
}

impl StreamTransformer for StreamPipeline {
    fn transform(&mut self, event: StreamEvent) -> Option<StreamEvent> {
        StreamPipeline::transform(self, event)
    }
}

// ---------------------------------------------------------------------------
// Helper: Transform a stream with a pipeline
// ---------------------------------------------------------------------------

/// Apply a stream pipeline to a boxed stream, returning a new transformed stream.
#[cfg(not(target_arch = "wasm32"))]
pub fn transform_stream(
    stream: crate::provider::BoxStream,
    mut pipeline: StreamPipeline,
) -> crate::provider::BoxStream {
    use futures::StreamExt;
    Box::new(stream.filter_map(move |result| {
        let output = match result {
            Ok(event) => pipeline.transform(event).map(Ok),
            Err(e) => Some(Err(e)),
        };
        std::future::ready(output)
    }))
}

/// Apply a stream pipeline to a boxed stream (WASM version).
#[cfg(target_arch = "wasm32")]
pub fn transform_stream(
    stream: crate::provider::BoxStream,
    mut pipeline: StreamPipeline,
) -> crate::provider::BoxStream {
    use futures::StreamExt;
    Box::new(stream.filter_map(move |result| {
        let output = match result {
            Ok(event) => pipeline.transform(event).map(Ok),
            Err(e) => Some(Err(e)),
        };
        std::future::ready(output)
    }))
}
