//! Cost estimation primitives for enterprise usage tracking.

use serde::{Deserialize, Serialize};

use crate::message::Usage;

#[derive(Debug, Clone, Copy)]
pub struct Pricing {
    pub input_per_million: f64,
    pub output_per_million: f64,
    pub reasoning_per_million: f64,
    pub cache_read_per_million: f64,
    pub cache_write_per_million: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEstimate {
    pub model: String,
    pub normalized_model: String,
    pub currency: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub reasoning_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_creation_tokens: u64,
    pub input_cost_usd: f64,
    pub output_cost_usd: f64,
    pub reasoning_cost_usd: f64,
    pub cache_read_cost_usd: f64,
    pub cache_creation_cost_usd: f64,
    pub total_cost_usd: f64,
}

const DEFAULT_PRICING: Pricing = Pricing {
    input_per_million: 1.0,
    output_per_million: 3.0,
    reasoning_per_million: 3.0,
    cache_read_per_million: 0.2,
    cache_write_per_million: 1.0,
};

const PRICING_TABLE: &[(&str, Pricing)] = &[
    (
        "gpt-5",
        Pricing {
            input_per_million: 1.25,
            output_per_million: 10.0,
            reasoning_per_million: 10.0,
            cache_read_per_million: 0.125,
            cache_write_per_million: 1.25,
        },
    ),
    (
        "gpt-4o",
        Pricing {
            input_per_million: 5.0,
            output_per_million: 15.0,
            reasoning_per_million: 15.0,
            cache_read_per_million: 2.5,
            cache_write_per_million: 5.0,
        },
    ),
    (
        "gpt-4.1",
        Pricing {
            input_per_million: 2.0,
            output_per_million: 8.0,
            reasoning_per_million: 8.0,
            cache_read_per_million: 1.0,
            cache_write_per_million: 2.0,
        },
    ),
    (
        "o4",
        Pricing {
            input_per_million: 1.1,
            output_per_million: 4.4,
            reasoning_per_million: 4.4,
            cache_read_per_million: 0.55,
            cache_write_per_million: 1.1,
        },
    ),
    (
        "o3",
        Pricing {
            input_per_million: 1.0,
            output_per_million: 4.0,
            reasoning_per_million: 4.0,
            cache_read_per_million: 0.5,
            cache_write_per_million: 1.0,
        },
    ),
    (
        "claude-opus-4",
        Pricing {
            input_per_million: 15.0,
            output_per_million: 75.0,
            reasoning_per_million: 75.0,
            cache_read_per_million: 1.5,
            cache_write_per_million: 15.0,
        },
    ),
    (
        "claude-sonnet-4",
        Pricing {
            input_per_million: 3.0,
            output_per_million: 15.0,
            reasoning_per_million: 15.0,
            cache_read_per_million: 0.3,
            cache_write_per_million: 3.0,
        },
    ),
    (
        "claude-haiku-4",
        Pricing {
            input_per_million: 0.8,
            output_per_million: 4.0,
            reasoning_per_million: 4.0,
            cache_read_per_million: 0.08,
            cache_write_per_million: 0.8,
        },
    ),
    (
        "gemini-2.5-pro",
        Pricing {
            input_per_million: 1.25,
            output_per_million: 5.0,
            reasoning_per_million: 5.0,
            cache_read_per_million: 0.125,
            cache_write_per_million: 1.25,
        },
    ),
    (
        "gemini-2.5-flash",
        Pricing {
            input_per_million: 0.3,
            output_per_million: 2.5,
            reasoning_per_million: 2.5,
            cache_read_per_million: 0.03,
            cache_write_per_million: 0.3,
        },
    ),
    (
        "deepseek-reasoner",
        Pricing {
            input_per_million: 0.55,
            output_per_million: 2.2,
            reasoning_per_million: 2.2,
            cache_read_per_million: 0.055,
            cache_write_per_million: 0.55,
        },
    ),
    (
        "deepseek-chat",
        Pricing {
            input_per_million: 0.27,
            output_per_million: 1.1,
            reasoning_per_million: 1.1,
            cache_read_per_million: 0.027,
            cache_write_per_million: 0.27,
        },
    ),
    (
        "llama-3.3",
        Pricing {
            input_per_million: 0.59,
            output_per_million: 0.79,
            reasoning_per_million: 0.79,
            cache_read_per_million: 0.059,
            cache_write_per_million: 0.59,
        },
    ),
    (
        "mixtral",
        Pricing {
            input_per_million: 0.6,
            output_per_million: 0.6,
            reasoning_per_million: 0.6,
            cache_read_per_million: 0.06,
            cache_write_per_million: 0.6,
        },
    ),
];

fn normalize_model(model: &str) -> String {
    let lower = model.trim().to_lowercase();
    if let Some((prefix, tail)) = lower.split_once('/')
        && matches!(
            prefix,
            "openai"
                | "anthropic"
                | "google"
                | "meta-llama"
                | "mistralai"
                | "deepseek"
                | "xai"
                | "cohere"
        )
        && !tail.is_empty()
    {
        return tail.to_string();
    }
    lower
}

fn resolve_pricing(model: &str) -> Pricing {
    let normalized = normalize_model(model);
    for (prefix, pricing) in PRICING_TABLE {
        if normalized.starts_with(prefix) {
            return *pricing;
        }
    }
    DEFAULT_PRICING
}

fn calc_usd(tokens: u64, per_million: f64) -> f64 {
    (tokens as f64 / 1_000_000.0) * per_million
}

pub fn estimate_cost(model: &str, usage: &Usage) -> CostEstimate {
    let pricing = resolve_pricing(model);
    let normalized_model = normalize_model(model);
    let reasoning_tokens = usage.reasoning_tokens.unwrap_or(0);
    let cache_read_tokens = usage.cache_read_tokens.unwrap_or(0);
    let cache_creation_tokens = usage.cache_creation_tokens.unwrap_or(0);

    let input_cost_usd = calc_usd(usage.input_tokens, pricing.input_per_million);
    let output_cost_usd = calc_usd(usage.output_tokens, pricing.output_per_million);
    let reasoning_cost_usd = calc_usd(reasoning_tokens, pricing.reasoning_per_million);
    let cache_read_cost_usd = calc_usd(cache_read_tokens, pricing.cache_read_per_million);
    let cache_creation_cost_usd = calc_usd(cache_creation_tokens, pricing.cache_write_per_million);
    let total_cost_usd = input_cost_usd
        + output_cost_usd
        + reasoning_cost_usd
        + cache_read_cost_usd
        + cache_creation_cost_usd;

    CostEstimate {
        model: model.to_string(),
        normalized_model,
        currency: "USD".to_string(),
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        reasoning_tokens,
        cache_read_tokens,
        cache_creation_tokens,
        input_cost_usd,
        output_cost_usd,
        reasoning_cost_usd,
        cache_read_cost_usd,
        cache_creation_cost_usd,
        total_cost_usd,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_cost_uses_known_pricing() {
        let usage = Usage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            ..Usage::default()
        };
        let estimate = estimate_cost("gpt-4o", &usage);
        assert_eq!(estimate.normalized_model, "gpt-4o");
        assert!(estimate.input_cost_usd > 0.0);
        assert!(estimate.output_cost_usd > 0.0);
        assert!(estimate.total_cost_usd >= estimate.input_cost_usd + estimate.output_cost_usd);
    }

    #[test]
    fn estimate_cost_normalizes_provider_prefixed_models() {
        let usage = Usage {
            input_tokens: 1000,
            output_tokens: 1000,
            ..Usage::default()
        };
        let estimate = estimate_cost("openai/gpt-5.2", &usage);
        assert_eq!(estimate.normalized_model, "gpt-5.2");
    }
}
