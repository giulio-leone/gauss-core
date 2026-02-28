use gauss_core::patterns::*;
use serde_json::json;

// ---- ToolValidator ----

#[test]
fn test_validator_null_to_default() {
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer", "default": 10}
        }
    });
    let input = json!({"name": "test"});
    let v = ToolValidator::new();
    let result = v.validate(input, &schema).unwrap();
    assert_eq!(result["count"], 10);
    assert_eq!(result["name"], "test");
}

#[test]
fn test_validator_type_cast_string_to_number() {
    let schema = json!({
        "type": "object",
        "properties": {
            "count": {"type": "number"}
        }
    });
    let input = json!({"count": "42.5"});
    let v = ToolValidator::new();
    let result = v.validate(input, &schema).unwrap();
    assert_eq!(result["count"], 42.5);
}

#[test]
fn test_validator_type_cast_string_to_bool() {
    let schema = json!({
        "type": "object",
        "properties": {
            "active": {"type": "boolean"}
        }
    });
    let input = json!({"active": "true"});
    let v = ToolValidator::new();
    let result = v.validate(input, &schema).unwrap();
    assert_eq!(result["active"], true);
}

#[test]
fn test_validator_json_parse() {
    let schema = json!({
        "type": "object",
        "properties": {
            "data": {"type": "object"}
        }
    });
    let input = json!({"data": r#"{"key": "value"}"#});
    let v = ToolValidator::new();
    let result = v.validate(input, &schema).unwrap();
    assert_eq!(result["data"]["key"], "value");
}

#[test]
fn test_validator_strip_null() {
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        }
    });
    let input = json!({"name": "test", "extra": null});
    let v = ToolValidator::new();
    let result = v.validate(input, &schema).unwrap();
    assert_eq!(result["name"], "test");
    assert!(result.get("extra").is_none());
}

#[test]
fn test_validator_combined_strategies() {
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer", "default": 5},
            "active": {"type": "boolean"}
        },
        "required": ["name"]
    });
    // count missing (gets default), active is string (gets cast)
    let input = json!({"name": "test", "active": "yes"});
    let v = ToolValidator::new();
    let result = v.validate(input, &schema).unwrap();
    assert_eq!(result["name"], "test");
    assert_eq!(result["count"], 5);
    assert_eq!(result["active"], true);
}

// ---- ToolChain ----

#[tokio::test]
async fn test_tool_chain_sequential() {
    use gauss_core::tool::Tool;
    use std::sync::Arc;

    let double = Tool::builder("double", "Double a number")
        .execute(|args: serde_json::Value| async move {
            let n = args["value"].as_f64().unwrap_or(0.0);
            Ok(json!({"value": n * 2.0}))
        })
        .build();

    let add_one = Tool::builder("add_one", "Add one")
        .execute(|args: serde_json::Value| async move {
            let n = args["value"].as_f64().unwrap_or(0.0);
            Ok(json!({"value": n + 1.0}))
        })
        .build();

    let chain = ToolChain::new("math_chain").add(double).add(add_one);

    assert_eq!(chain.len(), 2);
    assert!(!chain.is_empty());

    let result = chain.execute(json!({"value": 5.0})).await.unwrap();
    assert_eq!(result["value"], 11.0); // (5*2) + 1 = 11
}

#[tokio::test]
async fn test_tool_chain_traced() {
    use gauss_core::tool::Tool;

    let increment = Tool::builder("inc", "Increment")
        .execute(|args: serde_json::Value| async move {
            let n = args["n"].as_i64().unwrap_or(0);
            Ok(json!({"n": n + 1}))
        })
        .build();

    let chain = ToolChain::new("trace_test")
        .add(increment.clone())
        .add(increment);

    let steps = chain.execute_traced(json!({"n": 0})).await.unwrap();
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0].tool_name, "inc");
    assert_eq!(steps[0].output["n"], 1);
    assert_eq!(steps[1].output["n"], 2);
}

// ---- Plan ----

#[test]
fn test_plan_ready_steps() {
    let mut plan = Plan {
        goal: "Test".into(),
        steps: vec![
            PlanStep {
                id: "a".into(),
                description: "First".into(),
                depends_on: vec![],
                status: PlanStepStatus::Pending,
                result: None,
            },
            PlanStep {
                id: "b".into(),
                description: "Second".into(),
                depends_on: vec!["a".into()],
                status: PlanStepStatus::Pending,
                result: None,
            },
        ],
    };

    let ready = plan.ready_steps();
    assert_eq!(ready.len(), 1);
    assert_eq!(ready[0].id, "a");

    plan.set_status("a", PlanStepStatus::Done);
    let ready = plan.ready_steps();
    assert_eq!(ready.len(), 1);
    assert_eq!(ready[0].id, "b");
}

#[test]
fn test_plan_complete() {
    let mut plan = Plan {
        goal: "Test".into(),
        steps: vec![PlanStep {
            id: "a".into(),
            description: "Only step".into(),
            depends_on: vec![],
            status: PlanStepStatus::Pending,
            result: None,
        }],
    };

    assert!(!plan.is_complete());
    plan.set_status("a", PlanStepStatus::Done);
    assert!(plan.is_complete());
}

#[test]
fn test_plan_parse() {
    let json_text =
        r#"{"goal": "test", "steps": [{"id": "s1", "description": "do it", "depends_on": []}]}"#;
    let plan = PlanningAgent::parse_plan(json_text).unwrap();
    assert_eq!(plan.goal, "test");
    assert_eq!(plan.steps.len(), 1);
}

#[test]
fn test_plan_parse_from_markdown() {
    let text = r#"Here's my plan:
```json
{"goal": "build feature", "steps": [{"id": "s1", "description": "implement", "depends_on": []}]}
```"#;
    let plan = PlanningAgent::parse_plan(text).unwrap();
    assert_eq!(plan.goal, "build feature");
}
