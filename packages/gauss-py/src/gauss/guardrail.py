"""
GuardrailChain — Content moderation, PII detection, and input validation via Rust.
"""

from __future__ import annotations

import json
from typing import Any


class GuardrailChain:
    """
    Guardrail chain backed by Rust for content safety and validation.

    Example:
        >>> chain = GuardrailChain()
        >>> chain.add_content_moderation(threshold=0.8)
        >>> chain.add_pii_detection(action="mask")
        >>> chain.add_token_limit(max_tokens=4096)
    """

    def __init__(self) -> None:
        from gauss._native import create_guardrail_chain

        self._handle = create_guardrail_chain()

    def add_content_moderation(self, threshold: float = 0.7) -> "GuardrailChain":
        """Add content moderation guardrail."""
        from gauss._native import guardrail_chain_add_content_moderation

        guardrail_chain_add_content_moderation(self._handle, threshold)
        return self

    def add_pii_detection(self, action: str = "mask") -> "GuardrailChain":
        """Add PII detection guardrail. Actions: 'mask', 'block', 'warn'."""
        from gauss._native import guardrail_chain_add_pii_detection

        guardrail_chain_add_pii_detection(self._handle, action)
        return self

    def add_token_limit(self, max_tokens: int = 4096) -> "GuardrailChain":
        """Add token limit guardrail."""
        from gauss._native import guardrail_chain_add_token_limit

        guardrail_chain_add_token_limit(self._handle, max_tokens)
        return self

    def add_regex_filter(self, pattern: str, action: str = "block") -> "GuardrailChain":
        """Add regex filter guardrail. Actions: 'block', 'warn'."""
        from gauss._native import guardrail_chain_add_regex_filter

        guardrail_chain_add_regex_filter(self._handle, pattern, action)
        return self

    def add_schema(self, schema: dict[str, Any]) -> "GuardrailChain":
        """Add JSON schema validation guardrail."""
        from gauss._native import guardrail_chain_add_schema

        guardrail_chain_add_schema(self._handle, json.dumps(schema))
        return self

    def destroy(self) -> None:
        """Release the underlying Rust guardrail chain."""
        from gauss._native import destroy_guardrail_chain

        destroy_guardrail_chain(self._handle)

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass
