use crate::provider::PROVIDERS;
use crate::registry::{next_handle, HandleRegistry};
use gauss_core::{guardrail, middleware, resilience};
use gauss_core::provider::retry::RetryConfig;
use gauss_core::provider::Provider;
use napi::bindgen_prelude::*;
use std::sync::{Arc, Mutex};

// ============ Middleware ============

static MIDDLEWARE_CHAINS: HandleRegistry<Arc<Mutex<middleware::MiddlewareChain>>> =
    HandleRegistry::new();

#[napi]
pub fn create_middleware_chain() -> u32 {
    MIDDLEWARE_CHAINS.insert(Arc::new(Mutex::new(middleware::MiddlewareChain::new())))
}

#[napi]
pub fn middleware_use_logging(handle: u32) -> Result<()> {
    let chain = MIDDLEWARE_CHAINS.get_clone(handle)?;
    chain
        .lock()
        .unwrap()
        .use_middleware(Arc::new(middleware::LoggingMiddleware));
    Ok(())
}

#[napi]
pub fn middleware_use_caching(handle: u32, ttl_ms: u32) -> Result<()> {
    let chain = MIDDLEWARE_CHAINS.get_clone(handle)?;
    chain
        .lock()
        .unwrap()
        .use_middleware(Arc::new(middleware::CachingMiddleware::new(ttl_ms as u64)));
    Ok(())
}

#[napi]
pub fn destroy_middleware_chain(handle: u32) -> Result<()> {
    MIDDLEWARE_CHAINS.remove(handle)?;
    Ok(())
}

// ============ Guardrails ============

static GUARDRAIL_CHAINS: HandleRegistry<guardrail::GuardrailChain> = HandleRegistry::new();

#[napi]
pub fn create_guardrail_chain() -> u32 {
    GUARDRAIL_CHAINS.insert(guardrail::GuardrailChain::new())
}

#[napi]
pub fn guardrail_chain_add_content_moderation(
    handle: u32,
    block_patterns: Vec<String>,
    warn_patterns: Vec<String>,
) -> Result<()> {
    GUARDRAIL_CHAINS.with_mut(handle, |chain| {
        let mut g = guardrail::ContentModerationGuardrail::new();
        for p in block_patterns {
            g = g.block_pattern(&p, format!("Blocked pattern: {p}"));
        }
        for p in warn_patterns {
            g = g.warn_pattern(&p, format!("Warning pattern: {p}"));
        }
        chain.add(Arc::new(g));
        Ok(())
    })
}

#[napi]
pub fn guardrail_chain_add_pii_detection(handle: u32, action: String) -> Result<()> {
    let pii_action = match action.as_str() {
        "block" => guardrail::PiiAction::Block,
        "warn" => guardrail::PiiAction::Warn,
        "redact" => guardrail::PiiAction::Redact,
        _ => {
            return Err(napi::Error::from_reason(
                "Invalid PII action: block|warn|redact",
            ));
        }
    };
    GUARDRAIL_CHAINS.with_mut(handle, |chain| {
        chain.add(Arc::new(guardrail::PiiDetectionGuardrail::new(pii_action)));
        Ok(())
    })
}

#[napi]
pub fn guardrail_chain_add_token_limit(
    handle: u32,
    max_input: Option<u32>,
    max_output: Option<u32>,
) -> Result<()> {
    let mut g = guardrail::TokenLimitGuardrail::new();
    if let Some(m) = max_input {
        g = g.max_input(m as usize);
    }
    if let Some(m) = max_output {
        g = g.max_output(m as usize);
    }
    GUARDRAIL_CHAINS.with_mut(handle, |chain| {
        chain.add(Arc::new(g));
        Ok(())
    })
}

#[napi]
pub fn guardrail_chain_add_regex_filter(
    handle: u32,
    block_rules: Vec<String>,
    warn_rules: Vec<String>,
) -> Result<()> {
    let mut g = guardrail::RegexFilterGuardrail::new();
    for r in block_rules {
        g = g.block(&r, format!("Blocked by regex: {r}"));
    }
    for r in warn_rules {
        g = g.warn(&r, format!("Warning by regex: {r}"));
    }
    GUARDRAIL_CHAINS.with_mut(handle, |chain| {
        chain.add(Arc::new(g));
        Ok(())
    })
}

#[napi]
pub fn guardrail_chain_add_schema(handle: u32, schema_json: String) -> Result<()> {
    let schema: serde_json::Value = serde_json::from_str(&schema_json)
        .map_err(|e| napi::Error::from_reason(format!("Invalid JSON schema: {e}")))?;
    GUARDRAIL_CHAINS.with_mut(handle, |chain| {
        chain.add(Arc::new(guardrail::SchemaGuardrail::new(schema)));
        Ok(())
    })
}

#[napi]
pub fn guardrail_chain_list(handle: u32) -> Result<Vec<String>> {
    GUARDRAIL_CHAINS.with_mut(handle, |chain| {
        Ok(chain.list().into_iter().map(String::from).collect())
    })
}

#[napi]
pub fn destroy_guardrail_chain(handle: u32) -> Result<()> {
    GUARDRAIL_CHAINS.remove(handle)?;
    Ok(())
}

// ============ Resilience ============

#[napi]
pub fn create_fallback_provider(provider_handles: Vec<u32>) -> Result<u32> {
    let prov_reg = PROVIDERS.raw().lock().expect("registry mutex poisoned");
    let mut providers_vec: Vec<Arc<dyn Provider>> = Vec::new();
    for h in provider_handles {
        let p = prov_reg
            .get(&h)
            .ok_or_else(|| napi::Error::from_reason(format!("Provider {h} not found")))?
            .clone();
        providers_vec.push(p);
    }
    drop(prov_reg);

    let fallback = Arc::new(resilience::FallbackProvider::new(providers_vec));
    let id = next_handle();
    PROVIDERS
        .raw()
        .lock()
        .expect("registry mutex poisoned")
        .insert(id, fallback);
    Ok(id)
}

#[napi]
pub fn create_circuit_breaker(
    provider_handle: u32,
    failure_threshold: Option<u32>,
    recovery_timeout_ms: Option<u32>,
) -> Result<u32> {
    let inner = PROVIDERS
        .raw()
        .lock()
        .unwrap()
        .get(&provider_handle)
        .ok_or_else(|| napi::Error::from_reason("Provider not found"))?
        .clone();

    let config = resilience::CircuitBreakerConfig {
        failure_threshold: failure_threshold.unwrap_or(5),
        recovery_timeout_ms: recovery_timeout_ms.map(|v| v as u64).unwrap_or(30_000),
        success_threshold: 1,
    };

    let cb = Arc::new(resilience::CircuitBreaker::new(inner, config));
    let id = next_handle();
    PROVIDERS
        .raw()
        .lock()
        .expect("registry mutex poisoned")
        .insert(id, cb);
    Ok(id)
}

#[napi]
pub fn create_resilient_provider(
    primary_handle: u32,
    fallback_handles: Vec<u32>,
    enable_circuit_breaker: Option<bool>,
) -> Result<u32> {
    let prov_reg = PROVIDERS.raw().lock().expect("registry mutex poisoned");
    let primary = prov_reg
        .get(&primary_handle)
        .ok_or_else(|| napi::Error::from_reason("Primary provider not found"))?
        .clone();

    let mut builder = resilience::ResilientProviderBuilder::new(primary);
    builder = builder.retry(RetryConfig::default());

    if enable_circuit_breaker.unwrap_or(false) {
        builder = builder.circuit_breaker(resilience::CircuitBreakerConfig::default());
    }

    for h in &fallback_handles {
        let fb = prov_reg
            .get(h)
            .ok_or_else(|| napi::Error::from_reason(format!("Fallback provider {h} not found")))?
            .clone();
        builder = builder.fallback(fb);
    }
    drop(prov_reg);

    let provider = builder.build();
    let id = next_handle();
    PROVIDERS
        .raw()
        .lock()
        .expect("registry mutex poisoned")
        .insert(id, provider);
    Ok(id)
}
