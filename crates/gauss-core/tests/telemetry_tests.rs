use gauss_core::telemetry::*;

#[test]
fn test_span_builder() {
    let span = SpanBuilder::new("agent_run", SpanType::AgentRun)
        .attribute("model", serde_json::json!("gpt-4"))
        .finish();

    assert_eq!(span.name, "agent_run");
    assert!(matches!(span.span_type, SpanType::AgentRun));
    assert!(span.start_ms > 0);
    assert!(span.duration_ms > 0 || span.duration_ms == 0); // very fast
}

#[test]
fn test_span_builder_with_metadata() {
    let span = SpanBuilder::new("tool_call", SpanType::ToolCall)
        .attribute("tool_name", serde_json::json!("search"))
        .attribute("args", serde_json::json!({"q": "test"}))
        .finish();

    assert_eq!(span.attributes["tool_name"], "search");
    assert_eq!(span.attributes["args"]["q"], "test");
}

#[test]
fn test_telemetry_collector() {
    let collector = TelemetryCollector::new();

    let span1 = SpanBuilder::new("run1", SpanType::AgentRun).finish();
    let span2 = SpanBuilder::new("run2", SpanType::AgentRun).finish();

    collector.record_span(span1);
    collector.record_span(span2);

    let spans = collector.export_spans();
    assert_eq!(spans.len(), 2);
}

#[test]
fn test_agent_metrics() {
    let mut metrics = AgentMetrics::new();
    metrics.record_model_call(100, 500, 300);
    metrics.record_model_call(200, 600, 400);
    metrics.record_tool_call(50);

    assert_eq!(metrics.model_call_count, 2);
    assert_eq!(metrics.total_tool_calls, 1);
    assert_eq!(metrics.total_input_tokens, 1100);
    assert_eq!(metrics.total_output_tokens, 700);
}

#[test]
fn test_collector_clear() {
    let collector = TelemetryCollector::new();
    collector.record_span(SpanBuilder::new("s1", SpanType::AgentRun).finish());
    assert_eq!(collector.export_spans().len(), 1);

    collector.clear();
    assert_eq!(collector.export_spans().len(), 0);
}
