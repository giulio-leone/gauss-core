use gauss_core::eval::*;
use std::sync::Arc;

#[tokio::test]
async fn test_exact_match_scorer() {
    let scorer = ExactMatchScorer;

    // Exact match
    let score = scorer
        .score("What is 2+2?", "4", Some("4"), None)
        .await
        .unwrap();
    assert!((score - 1.0).abs() < 0.01);

    // Mismatch
    let score = scorer
        .score("What is 2+2?", "five", Some("4"), None)
        .await
        .unwrap();
    assert!((score - 0.0).abs() < 0.01);
}

#[tokio::test]
async fn test_contains_scorer() {
    let scorer = ContainsScorer;

    let score = scorer
        .score(
            "What language?",
            "Gauss is written in Rust",
            Some("Rust"),
            None,
        )
        .await
        .unwrap();
    assert!((score - 1.0).abs() < 0.01);

    let score = scorer
        .score(
            "What language?",
            "Gauss is written in Python",
            Some("Rust"),
            None,
        )
        .await
        .unwrap();
    assert!((score - 0.0).abs() < 0.01);
}

#[tokio::test]
async fn test_length_ratio_scorer() {
    let scorer = LengthRatioScorer;

    // Same length → score 1.0
    let score = scorer
        .score("Summarize", "world", Some("hello"), None)
        .await
        .unwrap();
    assert!((score - 1.0).abs() < 0.01);

    // Very different length → lower score
    let score = scorer
        .score(
            "Summarize",
            "a very long response that is much longer than expected",
            Some("hello"),
            None,
        )
        .await
        .unwrap();
    assert!(score < 0.5);
}

#[tokio::test]
async fn test_eval_runner() {
    let dataset = vec![
        EvalCase {
            id: "t1".into(),
            input: "2+2".into(),
            expected_output: Some("4".into()),
            context: None,
            metadata: Default::default(),
        },
        EvalCase {
            id: "t2".into(),
            input: "3+3".into(),
            expected_output: Some("6".into()),
            context: None,
            metadata: Default::default(),
        },
    ];

    let mut runner = EvalRunner::new();
    runner.add_scorer(Arc::new(ExactMatchScorer));

    let report = runner
        .run(&dataset, |case| {
            let expected = case.expected_output.clone().unwrap_or_default();
            Box::pin(async move { Ok(expected) })
        })
        .await
        .unwrap();

    assert_eq!(report.total, 2);
    assert_eq!(report.passed, 2);
    assert!((report.pass_rate() - 1.0).abs() < 0.01);
}

#[tokio::test]
async fn test_eval_runner_partial_pass() {
    let dataset = vec![
        EvalCase {
            id: "t1".into(),
            input: "q1".into(),
            expected_output: Some("correct".into()),
            context: None,
            metadata: Default::default(),
        },
        EvalCase {
            id: "t2".into(),
            input: "q2".into(),
            expected_output: Some("correct".into()),
            context: None,
            metadata: Default::default(),
        },
    ];

    let mut runner = EvalRunner::new();
    runner.add_scorer(Arc::new(ExactMatchScorer));

    let report = runner
        .run(&dataset, |case| {
            let input = case.input.clone();
            Box::pin(async move {
                if input == "q1" {
                    Ok("correct".to_string())
                } else {
                    Ok("wrong".to_string())
                }
            })
        })
        .await
        .unwrap();

    assert_eq!(report.total, 2);
    assert_eq!(report.passed, 1);
    assert_eq!(report.failed, 1);
}
