"""
Middleware — Compose guardrails, telemetry, and resilience into a single config.

NOTE: Rust PyO3 does not expose a generic middleware chain like NAPI does.
Instead, middleware in Python is composed from GuardrailChain + Telemetry + Resilience.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MiddlewareConfig:
    """Configuration for composing middleware from Rust primitives."""

    guardrail_handle: Optional[int] = None
    telemetry_handle: Optional[int] = None
    resilient_provider_handle: Optional[int] = None


def middleware(
    guardrail: Optional[dict[str, Any]] = None,
    telemetry: bool = False,
    resilience: Optional[dict[str, Any]] = None,
) -> MiddlewareConfig:
    """
    Create a middleware config from available Rust primitives.

    Args:
        guardrail: Guardrail config, e.g. {"content_moderation": {"block": ["harmful"]}}
        telemetry: Enable telemetry collection
        resilience: Resilience config, e.g. {"circuit_breaker": True, "fallbacks": [handle1, handle2]}

    Returns a MiddlewareConfig with handles to the created Rust resources.
    """
    config = MiddlewareConfig()

    if guardrail:
        from gauss._native import (
            create_guardrail_chain,
            guardrail_chain_add_content_moderation,
            guardrail_chain_add_pii_detection,
            guardrail_chain_add_token_limit,
        )

        handle = create_guardrail_chain()
        if "content_moderation" in guardrail:
            cm = guardrail["content_moderation"]
            guardrail_chain_add_content_moderation(
                handle,
                cm.get("block", []),
                cm.get("warn", []),
            )
        if "pii_detection" in guardrail:
            action = guardrail["pii_detection"].get("action", "redact")
            guardrail_chain_add_pii_detection(handle, action)
        if "token_limit" in guardrail:
            tl = guardrail["token_limit"]
            guardrail_chain_add_token_limit(
                handle,
                tl.get("max_input"),
                tl.get("max_output"),
            )
        config.guardrail_handle = handle

    if telemetry:
        from gauss._native import create_telemetry

        config.telemetry_handle = create_telemetry()

    return config
