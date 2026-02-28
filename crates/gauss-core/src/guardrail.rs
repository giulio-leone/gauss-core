//! Guardrails — input/output validation pipeline for agent safety.
//!
//! Provides a composable guardrail system that validates messages before
//! they reach the model and validates outputs before they reach the user.
//! Supports Block, Warn, Rewrite, and Allow actions with short-circuit.

use crate::error::{self, GaussError};
use crate::message::Message;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// GuardrailAction & Result
// ---------------------------------------------------------------------------

/// Action taken by a guardrail after validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum GuardrailAction {
    /// Content passes validation.
    Allow,
    /// Content is blocked — agent execution should stop.
    Block {
        reason: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        guardrail: Option<String>,
    },
    /// Content has a warning but can proceed.
    Warn {
        reason: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        guardrail: Option<String>,
    },
    /// Content should be rewritten before proceeding.
    Rewrite {
        original: String,
        rewritten: String,
        reason: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        guardrail: Option<String>,
    },
}

impl GuardrailAction {
    pub fn is_blocked(&self) -> bool {
        matches!(self, Self::Block { .. })
    }

    pub fn is_warning(&self) -> bool {
        matches!(self, Self::Warn { .. })
    }

    pub fn is_rewrite(&self) -> bool {
        matches!(self, Self::Rewrite { .. })
    }
}

/// Result of a guardrail validation.
#[derive(Debug, Clone)]
pub struct GuardrailResult {
    pub action: GuardrailAction,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl GuardrailResult {
    pub fn allow() -> Self {
        Self {
            action: GuardrailAction::Allow,
            metadata: HashMap::new(),
        }
    }

    pub fn block(reason: impl Into<String>) -> Self {
        Self {
            action: GuardrailAction::Block {
                reason: reason.into(),
                guardrail: None,
            },
            metadata: HashMap::new(),
        }
    }

    pub fn warn(reason: impl Into<String>) -> Self {
        Self {
            action: GuardrailAction::Warn {
                reason: reason.into(),
                guardrail: None,
            },
            metadata: HashMap::new(),
        }
    }

    pub fn rewrite(
        original: impl Into<String>,
        rewritten: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            action: GuardrailAction::Rewrite {
                original: original.into(),
                rewritten: rewritten.into(),
                reason: reason.into(),
                guardrail: None,
            },
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    fn with_guardrail_name(mut self, name: &str) -> Self {
        match &mut self.action {
            GuardrailAction::Block { guardrail, .. }
            | GuardrailAction::Warn { guardrail, .. }
            | GuardrailAction::Rewrite { guardrail, .. } => {
                *guardrail = Some(name.to_string());
            }
            GuardrailAction::Allow => {}
        }
        self
    }
}

// ---------------------------------------------------------------------------
// Guardrail Trait
// ---------------------------------------------------------------------------

/// A guardrail validates input messages and/or output text.
#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
pub trait Guardrail: Send + Sync {
    /// Unique name for this guardrail.
    fn name(&self) -> &str;

    /// Validate input messages before sending to the model.
    async fn validate_input(&self, _messages: &[Message]) -> error::Result<GuardrailResult> {
        Ok(GuardrailResult::allow())
    }

    /// Validate model output text before returning to the user.
    async fn validate_output(&self, _text: &str) -> error::Result<GuardrailResult> {
        Ok(GuardrailResult::allow())
    }
}

#[cfg(target_arch = "wasm32")]
#[async_trait(?Send)]
pub trait Guardrail {
    fn name(&self) -> &str;

    async fn validate_input(&self, _messages: &[Message]) -> error::Result<GuardrailResult> {
        Ok(GuardrailResult::allow())
    }

    async fn validate_output(&self, _text: &str) -> error::Result<GuardrailResult> {
        Ok(GuardrailResult::allow())
    }
}

// ---------------------------------------------------------------------------
// GuardrailChain
// ---------------------------------------------------------------------------

/// Ordered guardrail execution pipeline. Short-circuits on Block.
pub struct GuardrailChain {
    guardrails: Vec<crate::Shared<dyn Guardrail>>,
}

impl Default for GuardrailChain {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for GuardrailChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuardrailChain")
            .field("count", &self.guardrails.len())
            .finish()
    }
}

impl GuardrailChain {
    pub fn new() -> Self {
        Self {
            guardrails: Vec::new(),
        }
    }

    /// Add a guardrail to the chain.
    pub fn add(&mut self, guardrail: crate::Shared<dyn Guardrail>) {
        self.guardrails.push(guardrail);
    }

    /// Remove a guardrail by name.
    pub fn remove(&mut self, name: &str) -> bool {
        let before = self.guardrails.len();
        self.guardrails.retain(|g| g.name() != name);
        self.guardrails.len() < before
    }

    /// List guardrail names.
    pub fn list(&self) -> Vec<&str> {
        self.guardrails.iter().map(|g| g.name()).collect()
    }

    /// Validate input through all guardrails. Short-circuits on Block.
    /// Returns all results (including warnings) and whether execution was blocked.
    pub async fn validate_input(
        &self,
        messages: &[Message],
    ) -> error::Result<GuardrailChainResult> {
        let mut results = Vec::new();
        for guardrail in &self.guardrails {
            let result = guardrail
                .validate_input(messages)
                .await?
                .with_guardrail_name(guardrail.name());
            let blocked = result.action.is_blocked();
            results.push(result);
            if blocked {
                return Ok(GuardrailChainResult {
                    blocked: true,
                    results,
                });
            }
        }
        Ok(GuardrailChainResult {
            blocked: false,
            results,
        })
    }

    /// Validate output through all guardrails. Short-circuits on Block.
    pub async fn validate_output(&self, text: &str) -> error::Result<GuardrailChainResult> {
        let mut results = Vec::new();
        for guardrail in &self.guardrails {
            let result = guardrail
                .validate_output(text)
                .await?
                .with_guardrail_name(guardrail.name());
            let blocked = result.action.is_blocked();
            results.push(result);
            if blocked {
                return Ok(GuardrailChainResult {
                    blocked: true,
                    results,
                });
            }
        }
        Ok(GuardrailChainResult {
            blocked: false,
            results,
        })
    }
}

/// Aggregate result from a guardrail chain.
#[derive(Debug, Clone)]
pub struct GuardrailChainResult {
    pub blocked: bool,
    pub results: Vec<GuardrailResult>,
}

impl GuardrailChainResult {
    /// Get all warnings from the chain.
    pub fn warnings(&self) -> Vec<&GuardrailResult> {
        self.results
            .iter()
            .filter(|r| r.action.is_warning())
            .collect()
    }

    /// Get the block reason if blocked.
    pub fn block_reason(&self) -> Option<&str> {
        self.results.iter().find_map(|r| match &r.action {
            GuardrailAction::Block { reason, .. } => Some(reason.as_str()),
            _ => None,
        })
    }

    /// Apply all rewrites to the given text, returning the final version.
    pub fn apply_rewrites(&self, text: &str) -> String {
        let mut result = text.to_string();
        for r in &self.results {
            if let GuardrailAction::Rewrite {
                original,
                rewritten,
                ..
            } = &r.action
            {
                result = result.replace(original, rewritten);
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Built-in: ContentModeration
// ---------------------------------------------------------------------------

/// Pattern-based content moderation guardrail.
#[derive(Debug, Clone)]
pub struct ContentModerationGuardrail {
    blocked_patterns: Vec<(regex::Regex, String)>,
    warn_patterns: Vec<(regex::Regex, String)>,
}

impl ContentModerationGuardrail {
    pub fn new() -> Self {
        Self {
            blocked_patterns: Vec::new(),
            warn_patterns: Vec::new(),
        }
    }

    /// Add a pattern that blocks content.
    pub fn block_pattern(mut self, pattern: &str, reason: impl Into<String>) -> Self {
        if let Ok(re) = regex::Regex::new(pattern) {
            self.blocked_patterns.push((re, reason.into()));
        }
        self
    }

    /// Add a pattern that warns.
    pub fn warn_pattern(mut self, pattern: &str, reason: impl Into<String>) -> Self {
        if let Ok(re) = regex::Regex::new(pattern) {
            self.warn_patterns.push((re, reason.into()));
        }
        self
    }

    fn check_text(&self, text: &str) -> GuardrailResult {
        for (re, reason) in &self.blocked_patterns {
            if re.is_match(text) {
                return GuardrailResult::block(reason.clone());
            }
        }
        for (re, reason) in &self.warn_patterns {
            if re.is_match(text) {
                return GuardrailResult::warn(reason.clone());
            }
        }
        GuardrailResult::allow()
    }
}

impl Default for ContentModerationGuardrail {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl Guardrail for ContentModerationGuardrail {
    fn name(&self) -> &str {
        "content_moderation"
    }

    async fn validate_input(&self, messages: &[Message]) -> error::Result<GuardrailResult> {
        for msg in messages {
            if let Some(text) = msg.text() {
                let result = self.check_text(text);
                if !matches!(result.action, GuardrailAction::Allow) {
                    return Ok(result);
                }
            }
        }
        Ok(GuardrailResult::allow())
    }

    async fn validate_output(&self, text: &str) -> error::Result<GuardrailResult> {
        Ok(self.check_text(text))
    }
}

// ---------------------------------------------------------------------------
// Built-in: PiiDetection
// ---------------------------------------------------------------------------

/// Detects and optionally redacts PII (emails, phones, SSNs, credit cards).
#[derive(Debug, Clone)]
pub struct PiiDetectionGuardrail {
    action: PiiAction,
    patterns: Vec<(regex::Regex, &'static str, &'static str)>,
}

/// What to do when PII is detected.
#[derive(Debug, Clone, Copy)]
pub enum PiiAction {
    Block,
    Warn,
    Redact,
}

impl PiiDetectionGuardrail {
    pub fn new(action: PiiAction) -> Self {
        let patterns = vec![
            (
                regex::Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}").unwrap(),
                "email",
                "[EMAIL_REDACTED]",
            ),
            (
                regex::Regex::new(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b").unwrap(),
                "phone",
                "[PHONE_REDACTED]",
            ),
            (
                regex::Regex::new(r"\b\d{3}-\d{2}-\d{4}\b").unwrap(),
                "ssn",
                "[SSN_REDACTED]",
            ),
            (
                regex::Regex::new(r"\b(?:\d[ -]*?){13,16}\b").unwrap(),
                "credit_card",
                "[CC_REDACTED]",
            ),
        ];
        Self { action, patterns }
    }

    fn check_text(&self, text: &str) -> GuardrailResult {
        let mut detected: Vec<&str> = Vec::new();
        let mut redacted = text.to_string();

        for (re, pii_type, replacement) in &self.patterns {
            if re.is_match(text) {
                detected.push(pii_type);
                if matches!(self.action, PiiAction::Redact) {
                    redacted = re.replace_all(&redacted, *replacement).to_string();
                }
            }
        }

        if detected.is_empty() {
            return GuardrailResult::allow();
        }

        let reason = format!("PII detected: {}", detected.join(", "));

        match self.action {
            PiiAction::Block => GuardrailResult::block(reason),
            PiiAction::Warn => GuardrailResult::warn(reason),
            PiiAction::Redact => GuardrailResult::rewrite(text, redacted, reason),
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl Guardrail for PiiDetectionGuardrail {
    fn name(&self) -> &str {
        "pii_detection"
    }

    async fn validate_input(&self, messages: &[Message]) -> error::Result<GuardrailResult> {
        for msg in messages {
            if let Some(text) = msg.text() {
                let result = self.check_text(text);
                if !matches!(result.action, GuardrailAction::Allow) {
                    return Ok(result);
                }
            }
        }
        Ok(GuardrailResult::allow())
    }

    async fn validate_output(&self, text: &str) -> error::Result<GuardrailResult> {
        Ok(self.check_text(text))
    }
}

// ---------------------------------------------------------------------------
// Built-in: TokenLimit
// ---------------------------------------------------------------------------

/// Limits input/output tokens using the context module's counting.
#[derive(Debug, Clone)]
pub struct TokenLimitGuardrail {
    max_input_tokens: Option<usize>,
    max_output_tokens: Option<usize>,
}

impl TokenLimitGuardrail {
    pub fn new() -> Self {
        Self {
            max_input_tokens: None,
            max_output_tokens: None,
        }
    }

    pub fn max_input(mut self, tokens: usize) -> Self {
        self.max_input_tokens = Some(tokens);
        self
    }

    pub fn max_output(mut self, tokens: usize) -> Self {
        self.max_output_tokens = Some(tokens);
        self
    }
}

impl Default for TokenLimitGuardrail {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl Guardrail for TokenLimitGuardrail {
    fn name(&self) -> &str {
        "token_limit"
    }

    async fn validate_input(&self, messages: &[Message]) -> error::Result<GuardrailResult> {
        if let Some(max) = self.max_input_tokens {
            let total: usize = messages
                .iter()
                .filter_map(|m| m.text())
                .map(crate::context::count_tokens)
                .sum();
            if total > max {
                return Ok(GuardrailResult::block(format!(
                    "Input exceeds token limit: {total} > {max}"
                )));
            }
        }
        Ok(GuardrailResult::allow())
    }

    async fn validate_output(&self, text: &str) -> error::Result<GuardrailResult> {
        if let Some(max) = self.max_output_tokens {
            let count = crate::context::count_tokens(text);
            if count > max {
                return Ok(GuardrailResult::block(format!(
                    "Output exceeds token limit: {count} > {max}"
                )));
            }
        }
        Ok(GuardrailResult::allow())
    }
}

// ---------------------------------------------------------------------------
// Built-in: RegexFilter
// ---------------------------------------------------------------------------

/// Configurable regex-based filter with block/warn/rewrite actions.
#[derive(Debug, Clone)]
pub struct RegexFilterGuardrail {
    rules: Vec<RegexRule>,
}

#[derive(Debug, Clone)]
struct RegexRule {
    pattern: regex::Regex,
    action: RegexRuleAction,
}

#[derive(Debug, Clone)]
enum RegexRuleAction {
    Block(String),
    Warn(String),
    Rewrite(String, String),
}

impl RegexFilterGuardrail {
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    pub fn block(mut self, pattern: &str, reason: impl Into<String>) -> Self {
        if let Ok(re) = regex::Regex::new(pattern) {
            self.rules.push(RegexRule {
                pattern: re,
                action: RegexRuleAction::Block(reason.into()),
            });
        }
        self
    }

    pub fn warn(mut self, pattern: &str, reason: impl Into<String>) -> Self {
        if let Ok(re) = regex::Regex::new(pattern) {
            self.rules.push(RegexRule {
                pattern: re,
                action: RegexRuleAction::Warn(reason.into()),
            });
        }
        self
    }

    pub fn rewrite(
        mut self,
        pattern: &str,
        replacement: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        if let Ok(re) = regex::Regex::new(pattern) {
            self.rules.push(RegexRule {
                pattern: re,
                action: RegexRuleAction::Rewrite(replacement.into(), reason.into()),
            });
        }
        self
    }

    fn check_text(&self, text: &str) -> GuardrailResult {
        for rule in &self.rules {
            if rule.pattern.is_match(text) {
                return match &rule.action {
                    RegexRuleAction::Block(reason) => GuardrailResult::block(reason.clone()),
                    RegexRuleAction::Warn(reason) => GuardrailResult::warn(reason.clone()),
                    RegexRuleAction::Rewrite(replacement, reason) => {
                        let rewritten = rule
                            .pattern
                            .replace_all(text, replacement.as_str())
                            .to_string();
                        GuardrailResult::rewrite(text, rewritten, reason.clone())
                    }
                };
            }
        }
        GuardrailResult::allow()
    }
}

impl Default for RegexFilterGuardrail {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl Guardrail for RegexFilterGuardrail {
    fn name(&self) -> &str {
        "regex_filter"
    }

    async fn validate_input(&self, messages: &[Message]) -> error::Result<GuardrailResult> {
        for msg in messages {
            if let Some(text) = msg.text() {
                let result = self.check_text(text);
                if !matches!(result.action, GuardrailAction::Allow) {
                    return Ok(result);
                }
            }
        }
        Ok(GuardrailResult::allow())
    }

    async fn validate_output(&self, text: &str) -> error::Result<GuardrailResult> {
        Ok(self.check_text(text))
    }
}

// ---------------------------------------------------------------------------
// Built-in: SchemaGuardrail
// ---------------------------------------------------------------------------

/// Validates output against a JSON Schema.
#[derive(Debug, Clone)]
pub struct SchemaGuardrail {
    schema: serde_json::Value,
}

impl SchemaGuardrail {
    pub fn new(schema: serde_json::Value) -> Self {
        Self { schema }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl Guardrail for SchemaGuardrail {
    fn name(&self) -> &str {
        "schema"
    }

    async fn validate_output(&self, text: &str) -> error::Result<GuardrailResult> {
        let parsed: serde_json::Value = match serde_json::from_str(text) {
            Ok(v) => v,
            Err(e) => {
                return Ok(GuardrailResult::block(format!(
                    "Output is not valid JSON: {e}"
                )));
            }
        };

        let validator = match jsonschema::validator_for(&self.schema) {
            Ok(v) => v,
            Err(e) => {
                return Err(GaussError::internal(format!("Invalid schema: {e}")));
            }
        };

        match validator.validate(&parsed) {
            Ok(()) => Ok(GuardrailResult::allow()),
            Err(e) => Ok(GuardrailResult::block(format!(
                "Schema validation failed: {e}"
            ))),
        }
    }
}
