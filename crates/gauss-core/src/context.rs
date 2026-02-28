//! Context management — token counting, message pruning, summarization.
//!
//! Tracks context window usage and provides strategies for keeping
//! messages within model limits. Uses tiktoken-rs for precise counting
//! on native targets, falls back to approximation on WASM.

use crate::message::Message;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Token Counter
// ---------------------------------------------------------------------------

/// Model context window sizes (approximate).
pub fn context_window_size(model: &str) -> usize {
    match model {
        m if m.contains("gpt-4o") => 128_000,
        m if m.contains("gpt-4-turbo") => 128_000,
        m if m.contains("gpt-4") => 8_192,
        m if m.contains("gpt-3.5") => 16_385,
        m if m.contains("claude-3") => 200_000,
        m if m.contains("claude-4") => 200_000,
        m if m.contains("gemini") => 1_000_000,
        m if m.contains("deepseek") => 64_000,
        m if m.contains("llama") => 128_000,
        _ => 128_000, // conservative default
    }
}

/// Approximate token counter (4 chars ≈ 1 token).
pub fn count_tokens_approx(text: &str) -> usize {
    text.len().div_ceil(4)
}

/// Precise token counter using tiktoken (native only, cl100k_base encoding).
/// Falls back to approximation if tiktoken is unavailable.
#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
pub fn count_tokens(text: &str) -> usize {
    use std::sync::OnceLock;
    static BPE: OnceLock<Option<tiktoken_rs::CoreBPE>> = OnceLock::new();
    let bpe = BPE.get_or_init(|| tiktoken_rs::cl100k_base().ok());
    match bpe {
        Some(enc) => enc.encode_ordinary(text).len(),
        None => count_tokens_approx(text),
    }
}

/// On WASM, fall back to approximation.
#[cfg(any(not(feature = "native"), target_arch = "wasm32"))]
pub fn count_tokens(text: &str) -> usize {
    count_tokens_approx(text)
}

/// Count tokens for a specific model's encoding (native only).
/// Supports cl100k_base (GPT-4/3.5), o200k_base (GPT-4o), etc.
#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
pub fn count_tokens_for_model(text: &str, model: &str) -> usize {
    tiktoken_rs::get_bpe_from_model(model)
        .map(|enc| enc.encode_ordinary(text).len())
        .unwrap_or_else(|_| count_tokens(text))
}

#[cfg(any(not(feature = "native"), target_arch = "wasm32"))]
pub fn count_tokens_for_model(text: &str, _model: &str) -> usize {
    count_tokens_approx(text)
}

/// Count tokens in a message (includes role overhead).
pub fn count_message_tokens(msg: &Message) -> usize {
    let content_tokens = msg
        .content
        .iter()
        .map(|c| match c {
            crate::message::Content::Text { text } => count_tokens_approx(text),
            crate::message::Content::Image { .. } => 85,
            crate::message::Content::Audio { .. } => 100,
            crate::message::Content::ToolCall {
                name, arguments, ..
            } => count_tokens_approx(name) + count_tokens_approx(&arguments.to_string()),
            crate::message::Content::ToolResult { content, .. } => {
                count_tokens_approx(&content.to_string())
            }
            crate::message::Content::Reasoning { text } => count_tokens_approx(text),
            crate::message::Content::File { .. } => 50,
        })
        .sum::<usize>();
    content_tokens + 4
}

/// Count total tokens in a message list.
pub fn count_messages_tokens(messages: &[Message]) -> usize {
    messages.iter().map(count_message_tokens).sum::<usize>() + 3 // +3 for prompt formatting
}

// ---------------------------------------------------------------------------
// Context Window Tracker
// ---------------------------------------------------------------------------

/// Tracks current and maximum token usage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextTracker {
    pub model: String,
    pub max_tokens: usize,
    pub current_tokens: usize,
    /// Fraction of window to reserve for response (default 0.2).
    pub response_reserve: f64,
}

impl ContextTracker {
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            max_tokens: context_window_size(model),
            current_tokens: 0,
            response_reserve: 0.2,
        }
    }

    pub fn with_reserve(mut self, reserve: f64) -> Self {
        self.response_reserve = reserve;
        self
    }

    /// Available tokens for input (after reserving for response).
    pub fn available_tokens(&self) -> usize {
        let reserved = (self.max_tokens as f64 * self.response_reserve) as usize;
        self.max_tokens.saturating_sub(reserved)
    }

    /// Update current token count from messages.
    pub fn update(&mut self, messages: &[Message]) {
        self.current_tokens = count_messages_tokens(messages);
    }

    /// Check if messages exceed the available context window.
    pub fn is_over_limit(&self) -> bool {
        self.current_tokens > self.available_tokens()
    }

    /// How many tokens are over the limit.
    pub fn overflow(&self) -> usize {
        self.current_tokens.saturating_sub(self.available_tokens())
    }
}

// ---------------------------------------------------------------------------
// Pruning Strategies
// ---------------------------------------------------------------------------

/// Strategy for pruning messages when context is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PruningStrategy {
    /// Remove oldest non-system messages first.
    OldestFirst,
    /// Keep only the last N messages.
    SlidingWindow,
    /// Summarize old messages into a single system message.
    Summarize,
}

/// Configuration for context pruning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningConfig {
    pub strategy: PruningStrategy,
    /// For SlidingWindow: how many messages to keep.
    pub window_size: Option<usize>,
    /// Fraction of context window to trigger pruning (default 0.8).
    pub threshold: f64,
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self {
            strategy: PruningStrategy::OldestFirst,
            window_size: None,
            threshold: 0.8,
        }
    }
}

/// Prune messages to fit within context window.
pub fn prune_messages(messages: &[Message], model: &str, config: &PruningConfig) -> Vec<Message> {
    let max_tokens = context_window_size(model);
    let budget = (max_tokens as f64 * config.threshold) as usize;

    match config.strategy {
        PruningStrategy::OldestFirst => prune_oldest_first(messages, budget),
        PruningStrategy::SlidingWindow => {
            let window = config.window_size.unwrap_or(20);
            prune_sliding_window(messages, window)
        }
        PruningStrategy::Summarize => {
            // For summarization, we'd need an LLM call.
            // Fall back to oldest-first for now; real impl in middleware.
            prune_oldest_first(messages, budget)
        }
    }
}

fn prune_oldest_first(messages: &[Message], budget: usize) -> Vec<Message> {
    let system: Vec<&Message> = messages
        .iter()
        .filter(|m| m.role == crate::message::Role::System)
        .collect();
    let non_system: Vec<&Message> = messages
        .iter()
        .filter(|m| m.role != crate::message::Role::System)
        .collect();

    let system_tokens: usize = system.iter().map(|m| count_message_tokens(m)).sum();
    let remaining_budget = budget.saturating_sub(system_tokens);

    let mut kept: Vec<Message> = system.into_iter().cloned().collect();
    let mut used = 0;

    // Keep from the end (most recent)
    for msg in non_system.iter().rev() {
        let tokens = count_message_tokens(msg);
        if used + tokens > remaining_budget {
            break;
        }
        kept.push((*msg).clone());
        used += tokens;
    }

    // Restore order: system first, then recent messages
    let system_count = kept
        .iter()
        .filter(|m| m.role == crate::message::Role::System)
        .count();
    let mut result: Vec<Message> = kept[..system_count].to_vec();
    let mut recent: Vec<Message> = kept[system_count..].to_vec();
    recent.reverse();
    result.extend(recent);
    result
}

fn prune_sliding_window(messages: &[Message], window: usize) -> Vec<Message> {
    let system: Vec<Message> = messages
        .iter()
        .filter(|m| m.role == crate::message::Role::System)
        .cloned()
        .collect();
    let non_system: Vec<Message> = messages
        .iter()
        .filter(|m| m.role != crate::message::Role::System)
        .cloned()
        .collect();

    let start = non_system.len().saturating_sub(window);
    let mut result = system;
    result.extend(non_system[start..].to_vec());
    result
}
