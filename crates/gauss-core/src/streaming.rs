use serde::{Deserialize, Serialize};

use crate::message::Usage;
use crate::provider::FinishReason;

/// Events emitted during streaming generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// A text delta (partial token).
    TextDelta(String),

    /// A reasoning/thinking delta.
    ReasoningDelta(String),

    /// A partial tool call update.
    ToolCallDelta {
        index: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        arguments_delta: Option<String>,
    },

    /// Generation finished with a reason.
    FinishReason(FinishReason),

    /// Token usage statistics.
    Usage(Usage),

    /// Stream completed.
    Done,

    /// A step in a multi-step agent run completed.
    StepFinished {
        step_index: usize,
        finish_reason: FinishReason,
    },
}

impl StreamEvent {
    pub fn is_text_delta(&self) -> bool {
        matches!(self, Self::TextDelta(_))
    }

    pub fn is_done(&self) -> bool {
        matches!(self, Self::Done)
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::TextDelta(t) => Some(t),
            _ => None,
        }
    }
}
