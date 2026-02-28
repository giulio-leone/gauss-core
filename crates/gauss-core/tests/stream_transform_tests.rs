use gauss_core::stream_transform::*;
use gauss_core::streaming::StreamEvent;

#[test]
fn partial_json_complete_object() {
    let result = parse_partial_json(r#"{"name": "test", "age": 25}"#);
    assert!(result.is_some());
    let v = result.unwrap();
    assert_eq!(v["name"], "test");
    assert_eq!(v["age"], 25);
}

#[test]
fn partial_json_incomplete_object() {
    let result = parse_partial_json(r#"{"name": "test", "age": 2"#);
    assert!(result.is_some());
    let v = result.unwrap();
    assert_eq!(v["name"], "test");
    assert_eq!(v["age"], 2);
}

#[test]
fn partial_json_incomplete_string() {
    let result = parse_partial_json(r#"{"name": "te"#);
    assert!(result.is_some());
    let v = result.unwrap();
    assert_eq!(v["name"], "te");
}

#[test]
fn partial_json_nested() {
    let result = parse_partial_json(r#"{"user": {"name": "test""#);
    assert!(result.is_some());
    let v = result.unwrap();
    assert_eq!(v["user"]["name"], "test");
}

#[test]
fn partial_json_array() {
    let result = parse_partial_json(r#"[1, 2, 3"#);
    assert!(result.is_some());
    let v = result.unwrap();
    assert_eq!(v.as_array().unwrap().len(), 3);
}

#[test]
fn partial_json_empty() {
    let result = parse_partial_json("");
    assert!(result.is_none());
}

#[test]
fn object_accumulator_incremental() {
    let mut acc = ObjectAccumulator::new();

    // First chunk: closes to {} (valid empty object)
    let v = acc.feed("{");
    assert!(v.is_some());
    assert_eq!(v.unwrap(), serde_json::json!({}));

    // Second chunk: now has a key-value pair — different from {}
    let v = acc.feed(r#""name": "test""#);
    assert!(v.is_some());
    assert_eq!(v.unwrap()["name"], "test");

    // Same data, no change
    assert!(acc.feed("").is_none());

    // Final brace — same value as before, no change
    let v = acc.feed("}");
    assert!(v.is_none()); // Same parsed value, deduplicated

    assert_eq!(acc.text(), r#"{"name": "test"}"#);
}

#[test]
fn map_text_transformer() {
    let mut transformer = MapText::new(|text: &str| text.to_uppercase());

    let result = transformer.transform(StreamEvent::TextDelta("hello".into()));
    assert!(matches!(result, Some(StreamEvent::TextDelta(t)) if t == "HELLO"));

    // Non-text events pass through
    let result = transformer.transform(StreamEvent::Done);
    assert!(matches!(result, Some(StreamEvent::Done)));
}

#[test]
fn filter_events_transformer() {
    let mut transformer = FilterEvents::new(|event: &StreamEvent| event.is_text_delta());

    let result = transformer.transform(StreamEvent::TextDelta("hello".into()));
    assert!(result.is_some());

    let result = transformer.transform(StreamEvent::Done);
    assert!(result.is_none());
}

#[test]
fn tap_transformer() {
    use std::sync::atomic::{AtomicU32, Ordering};
    let counter = AtomicU32::new(0);
    let mut transformer = Tap::new(move |_event: &StreamEvent| {
        counter.fetch_add(1, Ordering::Relaxed);
    });

    transformer.transform(StreamEvent::TextDelta("hello".into()));
    transformer.transform(StreamEvent::Done);
    // Tap shouldn't filter anything
    let result = transformer.transform(StreamEvent::TextDelta("world".into()));
    assert!(result.is_some());
}

#[test]
fn object_delta_transformer() {
    let mut transformer = ObjectDeltaTransformer::new();

    // Start accumulating JSON
    let r1 = transformer.transform(StreamEvent::TextDelta(r#"{"name"#.into()));
    // May or may not parse yet - both are fine
    assert!(r1.is_some());

    let r2 = transformer.transform(StreamEvent::TextDelta(r#"": "test"}"#.into()));
    assert!(r2.is_some());
    // Should emit an ObjectDelta
    if let Some(StreamEvent::ObjectDelta(v)) = r2 {
        assert_eq!(v["name"], "test");
    }
}

#[test]
fn stream_pipeline_compose() {
    let mut pipeline = StreamPipeline::new()
        .pipe(Box::new(MapText::new(|t: &str| t.to_uppercase())))
        .pipe(Box::new(FilterEvents::new(|e: &StreamEvent| {
            e.is_text_delta()
        })));

    // Text delta: uppercased and kept
    let result = pipeline.transform(StreamEvent::TextDelta("hello".into()));
    assert!(matches!(result, Some(StreamEvent::TextDelta(t)) if t == "HELLO"));

    // Done event: filtered out
    let result = pipeline.transform(StreamEvent::Done);
    assert!(result.is_none());
}

#[test]
fn stream_pipeline_empty() {
    let mut pipeline = StreamPipeline::new();

    let result = pipeline.transform(StreamEvent::TextDelta("hello".into()));
    assert!(matches!(result, Some(StreamEvent::TextDelta(t)) if t == "hello"));
}
