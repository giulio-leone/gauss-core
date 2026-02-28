//! Evaluation framework — score agent responses, run benchmarks.
//!
//! Provides traits for scoring, built-in scorers, dataset loading,
//! and a batch evaluation runner.

use crate::error;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A single evaluation test case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalCase {
    pub id: String,
    pub input: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_output: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// The result of evaluating a single case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    pub case_id: String,
    pub scores: HashMap<String, f64>,
    pub passed: bool,
    pub actual_output: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub duration_ms: u64,
}

/// Aggregate results for a full evaluation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub errored: usize,
    pub avg_scores: HashMap<String, f64>,
    pub results: Vec<EvalResult>,
    pub total_duration_ms: u64,
}

impl EvalReport {
    /// Pass rate as a fraction (0.0–1.0).
    pub fn pass_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.passed as f64 / self.total as f64
    }
}

// ---------------------------------------------------------------------------
// Scorer Trait
// ---------------------------------------------------------------------------

/// Evaluates a response against expected criteria and returns a score (0.0–1.0).
#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
pub trait Scorer: Send + Sync {
    fn name(&self) -> &str;

    /// Score a response. Returns 0.0–1.0.
    async fn score(
        &self,
        input: &str,
        output: &str,
        expected: Option<&str>,
        context: Option<&str>,
    ) -> error::Result<f64>;
}

#[cfg(target_arch = "wasm32")]
#[async_trait(?Send)]
pub trait Scorer {
    fn name(&self) -> &str;

    async fn score(
        &self,
        input: &str,
        output: &str,
        expected: Option<&str>,
        context: Option<&str>,
    ) -> error::Result<f64>;
}

// ---------------------------------------------------------------------------
// Built-in Scorers
// ---------------------------------------------------------------------------

/// Exact match scorer — 1.0 if output matches expected, 0.0 otherwise.
pub struct ExactMatchScorer;

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl Scorer for ExactMatchScorer {
    fn name(&self) -> &str {
        "exact_match"
    }

    async fn score(
        &self,
        _input: &str,
        output: &str,
        expected: Option<&str>,
        _context: Option<&str>,
    ) -> error::Result<f64> {
        match expected {
            Some(exp) => Ok(if output.trim() == exp.trim() {
                1.0
            } else {
                0.0
            }),
            None => Ok(0.0),
        }
    }
}

/// Contains scorer — 1.0 if output contains expected, 0.0 otherwise.
pub struct ContainsScorer;

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl Scorer for ContainsScorer {
    fn name(&self) -> &str {
        "contains"
    }

    async fn score(
        &self,
        _input: &str,
        output: &str,
        expected: Option<&str>,
        _context: Option<&str>,
    ) -> error::Result<f64> {
        match expected {
            Some(exp) => Ok(if output.to_lowercase().contains(&exp.to_lowercase()) {
                1.0
            } else {
                0.0
            }),
            None => Ok(0.0),
        }
    }
}

/// Length ratio scorer — scores based on how close output length is to expected.
pub struct LengthRatioScorer;

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl Scorer for LengthRatioScorer {
    fn name(&self) -> &str {
        "length_ratio"
    }

    async fn score(
        &self,
        _input: &str,
        output: &str,
        expected: Option<&str>,
        _context: Option<&str>,
    ) -> error::Result<f64> {
        match expected {
            Some(exp) => {
                let out_len = output.len() as f64;
                let exp_len = exp.len() as f64;
                if exp_len == 0.0 {
                    return Ok(if out_len == 0.0 { 1.0 } else { 0.0 });
                }
                let ratio = out_len / exp_len;
                // Score: 1.0 at perfect match, degrades as ratio deviates from 1.0
                Ok((1.0 - (ratio - 1.0).abs()).max(0.0))
            }
            None => Ok(0.5), // No expected, default to 0.5
        }
    }
}

// ---------------------------------------------------------------------------
// Dataset Loading
// ---------------------------------------------------------------------------

/// Load eval cases from a JSONL string.
pub fn load_dataset_jsonl(jsonl: &str) -> error::Result<Vec<EvalCase>> {
    jsonl
        .lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
        .map(|(i, line)| {
            serde_json::from_str(line).map_err(|e| {
                error::GaussError::internal(format!("Failed to parse line {}: {}", i + 1, e))
            })
        })
        .collect()
}

/// Load eval cases from a JSON array string.
pub fn load_dataset_json(json: &str) -> error::Result<Vec<EvalCase>> {
    serde_json::from_str(json)
        .map_err(|e| error::GaussError::internal(format!("Failed to parse dataset: {}", e)))
}

// ---------------------------------------------------------------------------
// Eval Runner
// ---------------------------------------------------------------------------

/// Batch evaluation runner.
pub struct EvalRunner {
    scorers: Vec<crate::Shared<dyn Scorer>>,
    /// Minimum average score to pass (default 0.5).
    pub pass_threshold: f64,
}

impl Default for EvalRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl EvalRunner {
    pub fn new() -> Self {
        Self {
            scorers: Vec::new(),
            pass_threshold: 0.5,
        }
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.pass_threshold = threshold;
        self
    }

    pub fn add_scorer(&mut self, scorer: crate::Shared<dyn Scorer>) {
        self.scorers.push(scorer);
    }

    /// Evaluate a single case given the actual output.
    pub async fn evaluate_case(
        &self,
        case: &EvalCase,
        actual_output: &str,
    ) -> error::Result<EvalResult> {
        let start = std::time::Instant::now();
        let mut scores = HashMap::new();

        for scorer in &self.scorers {
            let score = scorer
                .score(
                    &case.input,
                    actual_output,
                    case.expected_output.as_deref(),
                    case.context.as_deref(),
                )
                .await?;
            scores.insert(scorer.name().to_string(), score);
        }

        let avg = if scores.is_empty() {
            0.0
        } else {
            scores.values().sum::<f64>() / scores.len() as f64
        };

        Ok(EvalResult {
            case_id: case.id.clone(),
            scores,
            passed: avg >= self.pass_threshold,
            actual_output: actual_output.to_string(),
            error: None,
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Run evaluation on a dataset with a provided evaluation function.
    pub async fn run<F, Fut>(&self, dataset: &[EvalCase], eval_fn: F) -> error::Result<EvalReport>
    where
        F: Fn(&EvalCase) -> Fut,
        Fut: std::future::Future<Output = error::Result<String>>,
    {
        let run_start = std::time::Instant::now();
        let mut results = Vec::new();
        let mut passed = 0;
        let mut failed = 0;
        let mut errored = 0;

        for case in dataset {
            match eval_fn(case).await {
                Ok(output) => {
                    let result = self.evaluate_case(case, &output).await?;
                    if result.passed {
                        passed += 1;
                    } else {
                        failed += 1;
                    }
                    results.push(result);
                }
                Err(e) => {
                    errored += 1;
                    results.push(EvalResult {
                        case_id: case.id.clone(),
                        scores: HashMap::new(),
                        passed: false,
                        actual_output: String::new(),
                        error: Some(e.to_string()),
                        duration_ms: 0,
                    });
                }
            }
        }

        // Compute average scores
        let mut avg_scores: HashMap<String, f64> = HashMap::new();
        let valid_results: Vec<&EvalResult> =
            results.iter().filter(|r| r.error.is_none()).collect();
        if !valid_results.is_empty() {
            for r in &valid_results {
                for (name, score) in &r.scores {
                    *avg_scores.entry(name.clone()).or_insert(0.0) += score;
                }
            }
            let count = valid_results.len() as f64;
            for v in avg_scores.values_mut() {
                *v /= count;
            }
        }

        Ok(EvalReport {
            total: dataset.len(),
            passed,
            failed,
            errored,
            avg_scores,
            results,
            total_duration_ms: run_start.elapsed().as_millis() as u64,
        })
    }
}
