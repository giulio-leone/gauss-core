use crate::registry::{py_err, HandleRegistry};
use gauss_core::{eval, telemetry};
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

// ============ Eval ============

static EVAL_RUNNERS: HandleRegistry<Arc<Mutex<eval::EvalRunner>>> = HandleRegistry::new();

#[pyfunction]
#[pyo3(signature = (threshold=None))]
pub fn create_eval_runner(threshold: Option<f64>) -> u32 {
    let mut runner = eval::EvalRunner::new();
    if let Some(t) = threshold {
        runner = runner.with_threshold(t);
    }
    EVAL_RUNNERS.insert(Arc::new(Mutex::new(runner)))
}

#[pyfunction]
pub fn eval_add_scorer(handle: u32, scorer_type: &str) -> PyResult<()> {
    let runner = EVAL_RUNNERS.get_clone(handle)?;
    let scorer: Arc<dyn eval::Scorer> = match scorer_type {
        "exact_match" => Arc::new(eval::ExactMatchScorer),
        "contains" => Arc::new(eval::ContainsScorer),
        "length_ratio" => Arc::new(eval::LengthRatioScorer),
        other => return Err(py_err(format!("Unknown scorer: {other}"))),
    };
    runner
        .lock()
        .expect("registry mutex poisoned")
        .add_scorer(scorer);
    Ok(())
}

#[pyfunction]
pub fn load_dataset_jsonl(jsonl: &str) -> PyResult<String> {
    let cases = eval::load_dataset_jsonl(jsonl).map_err(py_err)?;
    serde_json::to_string(&cases).map_err(py_err)
}

#[pyfunction]
pub fn load_dataset_json(json_str: &str) -> PyResult<String> {
    let cases = eval::load_dataset_json(json_str).map_err(py_err)?;
    serde_json::to_string(&cases).map_err(py_err)
}

#[pyfunction]
pub fn destroy_eval_runner(handle: u32) -> PyResult<()> {
    EVAL_RUNNERS.remove(handle)
}

// ============ Telemetry ============

static TELEMETRY_COLLECTORS: HandleRegistry<Arc<Mutex<telemetry::TelemetryCollector>>> =
    HandleRegistry::new();

#[pyfunction]
pub fn create_telemetry() -> u32 {
    TELEMETRY_COLLECTORS.insert(Arc::new(Mutex::new(telemetry::TelemetryCollector::new())))
}

#[pyfunction]
pub fn telemetry_record_span(handle: u32, span_json: &str) -> PyResult<()> {
    let coll = TELEMETRY_COLLECTORS.get_clone(handle)?;
    let span: telemetry::SpanRecord = serde_json::from_str(span_json).map_err(py_err)?;
    coll.lock()
        .expect("registry mutex poisoned")
        .record_span(span);
    Ok(())
}

#[pyfunction]
pub fn telemetry_export_spans(handle: u32) -> PyResult<String> {
    let coll = TELEMETRY_COLLECTORS.get_clone(handle)?;
    let spans = coll.lock().expect("registry mutex poisoned").export_spans();
    serde_json::to_string(&spans).map_err(py_err)
}

#[pyfunction]
pub fn telemetry_export_metrics(handle: u32) -> PyResult<String> {
    let coll = TELEMETRY_COLLECTORS.get_clone(handle)?;
    let metrics = coll
        .lock()
        .expect("registry mutex poisoned")
        .export_metrics();
    serde_json::to_string(&metrics).map_err(py_err)
}

#[pyfunction]
pub fn telemetry_clear(handle: u32) -> PyResult<()> {
    let coll = TELEMETRY_COLLECTORS.get_clone(handle)?;
    coll.lock().expect("registry mutex poisoned").clear();
    Ok(())
}

#[pyfunction]
pub fn destroy_telemetry(handle: u32) -> PyResult<()> {
    TELEMETRY_COLLECTORS.remove(handle)
}
