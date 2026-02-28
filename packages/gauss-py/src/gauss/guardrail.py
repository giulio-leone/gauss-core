"""
GuardrailChain — Content moderation, PII detection, and input validation via Rust.
Rust signatures match: block_patterns/warn_patterns for content_moderation, action for pii, max_input/max_output for token_limit.
"""

from __future__ import annotations

import json
from typing import Any, Optional


class GuardrailChain:
    """
    Guardrail chain backed by Rust for content safety and validation.

    Example:
        >>> chain = GuardrailChain()
        >>> chain.add_content_moderation(block_patterns=["harmful"], warn_patterns=["risky"])
        >>> chain.add_pii_detection(action="redact")
        >>> chain.add_token_limit(max_input=4096)
    """

    def __init__(self) -> None:
        from gauss._native import create_guardrail_chain

        self._handle = create_guardrail_chain()

    def add_content_moderation(
        self,
        block_patterns: Optional[list[str]] = None,
        warn_patterns: Optional[list[str]] = None,
    ) -> "GuardrailChain":
        """Add content moderation guardrail with block/warn pattern lists."""
        from gauss._native import guardrail_chain_add_content_moderation

        guardrail_chain_add_content_moderation(
            self._handle,
            block_patterns or [],
            warn_patterns or [],
        )
        return self

    def add_pii_detection(self, action: str = "redact") -> "GuardrailChain":
        """Add PII detection guardrail. Actions: 'block', 'warn', 'redact'."""
        from gauss._native import guardrail_chain_add_pii_detection

        guardrail_chain_add_pii_detection(self._handle, action)
        return self

    def add_token_limit(
        self,
        max_input: Optional[int] = None,
        max_output: Optional[int] = None,
    ) -> "GuardrailChain":
        """Add token limit guardrail."""
        from gauss._native import guardrail_chain_add_token_limit

        guardrail_chain_add_token_limit(self._handle, max_input, max_output)
        return self

    def add_regex_filter(
        self,
        block_rules: Optional[list[str]] = None,
        warn_rules: Optional[list[str]] = None,
    ) -> "GuardrailChain":
        """Add regex filter guardrail with block/warn rule lists."""
        from gauss._native import guardrail_chain_add_regex_filter

        guardrail_chain_add_regex_filter(
            self._handle,
            block_rules or [],
            warn_rules or [],
        )
        return self

    def add_schema(self, schema: dict[str, Any]) -> "GuardrailChain":
        """Add JSON schema validation guardrail."""
        from gauss._native import guardrail_chain_add_schema

        guardrail_chain_add_schema(self._handle, json.dumps(schema))
        return self

    def list(self) -> list[str]:
        """List all guardrails in the chain."""
        from gauss._native import guardrail_chain_list

        return guardrail_chain_list(self._handle)

    def destroy(self) -> None:
        """Release the underlying Rust guardrail chain."""
        from gauss._native import destroy_guardrail_chain

        destroy_guardrail_chain(self._handle)

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass
