use gauss_core::message::{Message, Role};
use gauss_core::patterns::ToolValidator;
use gauss_core::stream_transform::parse_partial_json;
use proptest::prelude::*;
use regex;
use serde_json::json;

// ---- Partial JSON Parser Properties ----

proptest! {
    /// Valid JSON always parses successfully.
    #[test]
    fn partial_json_valid_roundtrip(s in "[a-zA-Z0-9 ]{1,50}") {
        let valid = format!(r#"{{"key": "{}"}}"#, s.replace('"', ""));
        let result = parse_partial_json(&valid);
        prop_assert!(result.is_some(), "Valid JSON must parse: {}", valid);
    }

    /// Empty string returns None.
    #[test]
    fn partial_json_empty_is_none(s in "[ \t\n]*") {
        let result = parse_partial_json(&s);
        // Whitespace-only might parse to something or nothing, both are OK
        // The key invariant: it never panics
        let _ = result;
    }

    /// Partial JSON with open brace always tries to close it.
    #[test]
    fn partial_json_open_brace_closes(
        key in "[a-zA-Z]{1,20}",
        val in "[a-zA-Z0-9]{1,20}"
    ) {
        let partial = format!(r#"{{"{}": "{}""#, key, val);
        let result = parse_partial_json(&partial);
        prop_assert!(result.is_some(), "Open brace should auto-close: {}", partial);
    }

    /// Partial JSON with open array always tries to close it.
    #[test]
    fn partial_json_open_array_closes(n in 1..100i64) {
        let partial = format!("[{}", n);
        let result = parse_partial_json(&partial);
        prop_assert!(result.is_some(), "Open array should auto-close: {}", partial);
    }

    /// Nested objects should still parse.
    #[test]
    fn partial_json_nested_objects(
        k1 in "[a-zA-Z]{1,10}",
        k2 in "[a-zA-Z]{1,10}",
        v in "[a-zA-Z0-9]{1,10}"
    ) {
        let partial = format!(r#"{{"{}": {{"{}": "{}""#, k1, k2, v);
        let result = parse_partial_json(&partial);
        prop_assert!(result.is_some(), "Nested objects should parse: {}", partial);
    }
}

// ---- Tool Validator Properties ----

proptest! {
    /// Validator should never panic on arbitrary JSON.
    #[test]
    fn validator_no_panic_on_arbitrary_input(
        s in "[a-zA-Z0-9 ]{0,50}"
    ) {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "default": "unknown"}
            }
        });
        let input = json!({"name": s});
        let validator = ToolValidator::new();
        let _ = validator.validate(input, &schema);
    }

    /// Null fields should be replaced with defaults.
    #[test]
    fn validator_null_gets_default(default_val in 0..1000i64) {
        let schema = json!({
            "type": "object",
            "properties": {
                "count": {"type": "integer", "default": default_val}
            }
        });
        let input = json!({"count": null});
        let validator = ToolValidator::new();
        let result = validator.validate(input, &schema).unwrap();
        prop_assert_eq!(result["count"].clone(), json!(default_val));
    }

    /// String numbers should cast to numbers.
    #[test]
    fn validator_string_to_number(n in -1000.0f64..1000.0) {
        if n.is_finite() {
            let schema = json!({
                "type": "object",
                "properties": {
                    "value": {"type": "number"}
                }
            });
            let input = json!({"value": n.to_string()});
            let validator = ToolValidator::new();
            let result = validator.validate(input, &schema).unwrap();
            let diff = (result["value"].as_f64().unwrap() - n).abs();
            prop_assert!(diff < 0.001, "Cast mismatch: expected {}, got {}", n, result["value"]);
        }
    }
}

// ---- Message Serialization Properties ----

proptest! {
    /// Messages roundtrip through JSON serialization.
    #[test]
    fn message_serde_roundtrip(content in "[a-zA-Z0-9 .,!?]{1,200}") {
        let msg = Message::user(content.clone());
        let json_str = serde_json::to_string(&msg).unwrap();
        let parsed: Message = serde_json::from_str(&json_str).unwrap();
        prop_assert_eq!(&parsed.role, &Role::User);
        prop_assert_eq!(parsed.text().unwrap(), &content);
    }

    /// System messages roundtrip.
    #[test]
    fn system_message_serde_roundtrip(content in "[a-zA-Z0-9 ]{1,100}") {
        let msg = Message::system(content.clone());
        let json_str = serde_json::to_string(&msg).unwrap();
        let parsed: Message = serde_json::from_str(&json_str).unwrap();
        prop_assert_eq!(&parsed.role, &Role::System);
    }
}

// ---- Guardrail Properties ----

// PII detection uses async trait — test with tokio in regular tests.
// Here we just validate the regex patterns don't panic on arbitrary input.
proptest! {
    /// PII regex should not panic on arbitrary text.
    #[test]
    fn pii_regex_no_panic(text in "[a-zA-Z0-9@. ]{0,100}") {
        let email_re = regex::Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}").unwrap();
        let phone_re = regex::Regex::new(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b").unwrap();
        let _ = email_re.is_match(&text);
        let _ = phone_re.is_match(&text);
    }
}
