use criterion::{Criterion, black_box, criterion_group, criterion_main};
use gauss_core::message::Message;
use gauss_core::patterns::ToolValidator;
use gauss_core::stream_transform::parse_partial_json;
use serde_json::json;

fn bench_partial_json_small(c: &mut Criterion) {
    c.bench_function("partial_json_small", |b| {
        b.iter(|| {
            parse_partial_json(black_box(r#"{"name": "test", "age": 2"#));
        });
    });
}

fn bench_partial_json_large(c: &mut Criterion) {
    let large = r#"{"users": [{"id": 1, "name": "Alice", "email": "alice@test.com"}, {"id": 2, "name": "Bob", "email": "bob@test.com"}, {"id": 3, "name": "Charlie"#;
    c.bench_function("partial_json_large", |b| {
        b.iter(|| {
            parse_partial_json(black_box(large));
        });
    });
}

fn bench_tool_validation(c: &mut Criterion) {
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer", "default": 10},
            "active": {"type": "boolean"}
        },
        "required": ["name"]
    });
    let input = json!({"name": "test", "count": "42", "active": "true"});

    c.bench_function("tool_validation", |b| {
        let validator = ToolValidator::new();
        b.iter(|| {
            validator
                .validate(black_box(input.clone()), black_box(&schema))
                .unwrap();
        });
    });
}

fn bench_message_creation(c: &mut Criterion) {
    c.bench_function("message_create_user", |b| {
        b.iter(|| {
            Message::user(black_box(
                "Hello, world! This is a test message.".to_string(),
            ));
        });
    });
}

fn bench_message_serialization(c: &mut Criterion) {
    let msg = Message::user("Hello, world! This is a test message.".to_string());
    c.bench_function("message_serialize", |b| {
        b.iter(|| {
            serde_json::to_string(black_box(&msg)).unwrap();
        });
    });
}

fn bench_guardrail_chain(c: &mut Criterion) {
    use gauss_core::guardrail::*;
    use std::sync::Arc;

    let mut chain = GuardrailChain::new();
    chain.add(Arc::new(
        ContentModerationGuardrail::new().block_pattern("blocked", "test"),
    ));
    chain.add(Arc::new(PiiDetectionGuardrail::new(PiiAction::Warn)));
    chain.add(Arc::new(TokenLimitGuardrail::new().max_input(4096)));

    let messages = vec![Message::user(
        "Hello, this is a normal message without any issues.".to_string(),
    )];

    c.bench_function("guardrail_chain_3_guards", |b| {
        b.iter(|| {
            chain.validate_input(black_box(&messages));
        });
    });
}

criterion_group!(
    benches,
    bench_partial_json_small,
    bench_partial_json_large,
    bench_tool_validation,
    bench_message_creation,
    bench_message_serialization,
    bench_guardrail_chain,
);
criterion_main!(benches);
