use crate::registry::HandleRegistry;
use gauss_core::{eval, telemetry};
use napi::bindgen_prelude::*;
use std::sync::{Arc, Mutex};

// ============ Eval ============

static EVAL_RUNNERS: HandleRegistry<Arc<Mutex<eval::EvalRunner>>> = HandleRegistry::new();

#[napi]
pub fn create_eval_runner(threshold: Option<f64>) -> u32 {
    let mut runner = eval::EvalRunner::new();
    if let Some(t) = threshold {
        runner = runner.with_threshold(t);
    }
    EVAL_RUNNERS.insert(Arc::new(Mutex::new(runner)))
}

#[napi]
pub fn eval_add_scorer(handle: u32, scorer_type: String) -> Result<()> {
    let runner = EVAL_RUNNERS.get_clone(handle)?;
    let scorer: Arc<dyn eval::Scorer> = match scorer_type.as_str() {
        "exact_match" => Arc::new(eval::ExactMatchScorer),
        "contains" => Arc::new(eval::ContainsScorer),
        "length_ratio" => Arc::new(eval::LengthRatioScorer),
        other => {
            return Err(napi::Error::from_reason(format!(
                "Unknown scorer: {other}. Use: exact_match, contains, length_ratio"
            )));
        }
    };
    runner.lock().expect("mutex poisoned").add_scorer(scorer);
    Ok(())
}

#[napi]
pub fn load_dataset_jsonl(jsonl: String) -> Result<serde_json::Value> {
    let cases = eval::load_dataset_jsonl(&jsonl)
        .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
    serde_json::to_value(&cases).map_err(|e| napi::Error::from_reason(format!("{e}")))
}

#[napi]
pub fn load_dataset_json(json_str: String) -> Result<serde_json::Value> {
    let cases = eval::load_dataset_json(&json_str)
        .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
    serde_json::to_value(&cases).map_err(|e| napi::Error::from_reason(format!("{e}")))
}

#[napi]
pub fn destroy_eval_runner(handle: u32) -> Result<()> {
    EVAL_RUNNERS.remove(handle)
}

// ============ Telemetry ============

static TELEMETRY_COLLECTORS: HandleRegistry<Arc<Mutex<telemetry::TelemetryCollector>>> =
    HandleRegistry::new();

#[napi]
pub fn create_telemetry() -> u32 {
    TELEMETRY_COLLECTORS.insert(Arc::new(Mutex::new(telemetry::TelemetryCollector::new())))
}

#[napi]
pub fn telemetry_record_span(handle: u32, span_json: String) -> Result<()> {
    let collector = TELEMETRY_COLLECTORS.get_clone(handle)?;
    let span: telemetry::SpanRecord = serde_json::from_str(&span_json)
        .map_err(|e| napi::Error::from_reason(format!("Invalid span JSON: {e}")))?;
    collector.lock().expect("mutex poisoned").record_span(span);
    Ok(())
}

#[napi]
pub fn telemetry_export_spans(handle: u32) -> Result<serde_json::Value> {
    let collector = TELEMETRY_COLLECTORS.get_clone(handle)?;
    let spans = collector.lock().expect("mutex poisoned").export_spans();
    serde_json::to_value(&spans)
        .map_err(|e| napi::Error::from_reason(format!("Serialize error: {e}")))
}

#[napi]
pub fn telemetry_export_metrics(handle: u32) -> Result<serde_json::Value> {
    let collector = TELEMETRY_COLLECTORS.get_clone(handle)?;
    let metrics = collector.lock().expect("mutex poisoned").export_metrics();
    serde_json::to_value(&metrics)
        .map_err(|e| napi::Error::from_reason(format!("Serialize error: {e}")))
}

#[napi]
pub fn telemetry_clear(handle: u32) -> Result<()> {
    let collector = TELEMETRY_COLLECTORS.get_clone(handle)?;
    collector.lock().expect("mutex poisoned").clear();
    Ok(())
}

#[napi]
pub fn destroy_telemetry(handle: u32) -> Result<()> {
    TELEMETRY_COLLECTORS.remove(handle)
}
