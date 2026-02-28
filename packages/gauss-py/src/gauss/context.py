"""
Context — Token counting and context window utilities via Rust.
"""

from __future__ import annotations

import json
from typing import Any


def count_tokens(text: str) -> int:
    """Count tokens in text using the default tokenizer."""
    from gauss._native import count_tokens as _count

    return _count(text)


def count_tokens_for_model(text: str, model: str) -> int:
    """Count tokens for a specific model."""
    from gauss._native import count_tokens_for_model as _count

    return _count(text, model)


def count_message_tokens(messages: list[dict[str, str]]) -> int:
    """Count tokens across a list of messages."""
    from gauss._native import count_message_tokens as _count

    return _count(json.dumps(messages))


def get_context_window_size(model: str) -> int:
    """Get the context window size for a model."""
    from gauss._native import get_context_window_size as _get

    return _get(model)
