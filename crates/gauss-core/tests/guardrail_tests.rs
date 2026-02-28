use gauss_core::guardrail::*;
use gauss_core::message::Message;

#[tokio::test]
async fn guardrail_allow() {
    let chain = GuardrailChain::new();
    let result = chain
        .validate_input(&[Message::user("hello")])
        .await
        .unwrap();
    assert!(!result.blocked);
    assert!(result.results.is_empty());
}

#[tokio::test]
async fn content_moderation_block() {
    let guardrail = ContentModerationGuardrail::new()
        .block_pattern(r"(?i)hack\s*system", "Hacking attempt detected");

    let result = guardrail
        .validate_input(&[Message::user("hack system now")])
        .await
        .unwrap();
    assert!(result.action.is_blocked());
}

#[tokio::test]
async fn content_moderation_warn() {
    let guardrail =
        ContentModerationGuardrail::new().warn_pattern(r"(?i)password", "Sensitive topic detected");

    let result = guardrail
        .validate_input(&[Message::user("my password is 123")])
        .await
        .unwrap();
    assert!(result.action.is_warning());
}

#[tokio::test]
async fn content_moderation_allow() {
    let guardrail = ContentModerationGuardrail::new().block_pattern(r"(?i)hack", "blocked");

    let result = guardrail
        .validate_input(&[Message::user("hello world")])
        .await
        .unwrap();
    assert!(matches!(result.action, GuardrailAction::Allow));
}

#[tokio::test]
async fn pii_detection_block() {
    let guardrail = PiiDetectionGuardrail::new(PiiAction::Block);

    let result = guardrail
        .validate_input(&[Message::user("my email is test@example.com")])
        .await
        .unwrap();
    assert!(result.action.is_blocked());
}

#[tokio::test]
async fn pii_detection_redact() {
    let guardrail = PiiDetectionGuardrail::new(PiiAction::Redact);

    let result = guardrail
        .validate_output("Contact me at test@example.com")
        .await
        .unwrap();
    assert!(result.action.is_rewrite());
    if let GuardrailAction::Rewrite { rewritten, .. } = &result.action {
        assert!(rewritten.contains("[EMAIL_REDACTED]"));
        assert!(!rewritten.contains("test@example.com"));
    }
}

#[tokio::test]
async fn pii_detection_phone() {
    let guardrail = PiiDetectionGuardrail::new(PiiAction::Warn);

    let result = guardrail
        .validate_output("Call 555-123-4567")
        .await
        .unwrap();
    assert!(result.action.is_warning());
}

#[tokio::test]
async fn token_limit_input_block() {
    let guardrail = TokenLimitGuardrail::new().max_input(5);

    let result = guardrail
        .validate_input(&[Message::user(
            "This is a very long message that exceeds the token limit by a lot",
        )])
        .await
        .unwrap();
    assert!(result.action.is_blocked());
}

#[tokio::test]
async fn token_limit_input_allow() {
    let guardrail = TokenLimitGuardrail::new().max_input(1000);

    let result = guardrail
        .validate_input(&[Message::user("short")])
        .await
        .unwrap();
    assert!(matches!(result.action, GuardrailAction::Allow));
}

#[tokio::test]
async fn regex_filter_block() {
    let guardrail = RegexFilterGuardrail::new().block(r"(?i)drop\s+table", "SQL injection attempt");

    let result = guardrail
        .validate_input(&[Message::user("DROP TABLE users")])
        .await
        .unwrap();
    assert!(result.action.is_blocked());
}

#[tokio::test]
async fn regex_filter_rewrite() {
    let guardrail =
        RegexFilterGuardrail::new().rewrite(r"(?i)badword", "***", "Profanity filtered");

    let result = guardrail
        .validate_output("this has a badword in it")
        .await
        .unwrap();
    assert!(result.action.is_rewrite());
    if let GuardrailAction::Rewrite { rewritten, .. } = &result.action {
        assert_eq!(rewritten, "this has a *** in it");
    }
}

#[tokio::test]
async fn schema_guardrail_valid() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" }
        },
        "required": ["name"]
    });
    let guardrail = SchemaGuardrail::new(schema);

    let result = guardrail
        .validate_output(r#"{"name": "test"}"#)
        .await
        .unwrap();
    assert!(matches!(result.action, GuardrailAction::Allow));
}

#[tokio::test]
async fn schema_guardrail_invalid() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" }
        },
        "required": ["name"]
    });
    let guardrail = SchemaGuardrail::new(schema);

    let result = guardrail.validate_output(r#"{"age": 25}"#).await.unwrap();
    assert!(result.action.is_blocked());
}

#[tokio::test]
async fn guardrail_chain_short_circuit() {
    use std::sync::Arc;

    let mut chain = GuardrailChain::new();
    chain.add(Arc::new(
        ContentModerationGuardrail::new().block_pattern(r"(?i)blocked", "Content blocked"),
    ));
    chain.add(Arc::new(
        RegexFilterGuardrail::new().warn(r".*", "This should not be reached"),
    ));

    let result = chain
        .validate_input(&[Message::user("this is blocked")])
        .await
        .unwrap();
    assert!(result.blocked);
    assert_eq!(result.results.len(), 1); // short-circuited, second guardrail not reached
}

#[tokio::test]
async fn guardrail_chain_apply_rewrites() {
    use std::sync::Arc;

    let mut chain = GuardrailChain::new();
    chain.add(Arc::new(
        RegexFilterGuardrail::new().rewrite(r"bad", "good", "cleaned"),
    ));

    let result = chain.validate_output("this is bad stuff").await.unwrap();
    assert!(!result.blocked);
    let cleaned = result.apply_rewrites("this is bad stuff");
    assert_eq!(cleaned, "this is good stuff");
}

#[tokio::test]
async fn guardrail_chain_list() {
    use std::sync::Arc;

    let mut chain = GuardrailChain::new();
    chain.add(Arc::new(ContentModerationGuardrail::new()));
    chain.add(Arc::new(PiiDetectionGuardrail::new(PiiAction::Block)));

    let names = chain.list();
    assert_eq!(names, vec!["content_moderation", "pii_detection"]);
}
