"""
Middleware — Native Rust middleware configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MiddlewareConfig:
    """Configuration for native Rust middleware."""

    logging: bool = False
    caching: dict[str, Any] | None = None
    guardrail: dict[str, Any] | None = None
    telemetry: bool = False

    # Internal handles
    _chain_handle: int | None = None
    _guardrail_handle: int | None = None
    _telemetry_handle: int | None = None

    def destroy(self) -> None:
        """Release all native middleware resources."""
        from gauss._native import (
            destroy_middleware_chain,
            destroy_guardrail_chain,
            destroy_telemetry,
        )

        if self._chain_handle is not None:
            destroy_middleware_chain(self._chain_handle)
        if self._guardrail_handle is not None:
            destroy_guardrail_chain(self._guardrail_handle)
        if self._telemetry_handle is not None:
            destroy_telemetry(self._telemetry_handle)


def middleware(
    logging: bool = False,
    caching: dict[str, Any] | None = None,
    guardrail: dict[str, Any] | None = None,
    telemetry: bool = False,
) -> MiddlewareConfig:
    """
    Create a native Rust middleware configuration.

    Args:
        logging: Enable logging middleware
        caching: Cache config, e.g. {"ttl_ms": 60_000}
        guardrail: Guardrail config, e.g. {"content_moderation": {"threshold": 0.8}}
        telemetry: Enable telemetry collection

    Example:
        >>> mw = middleware(logging=True, caching={"ttl_ms": 60_000})
    """
    from gauss._native import (
        create_middleware_chain,
        middleware_use_logging,
        middleware_use_caching,
        create_guardrail_chain,
        guardrail_chain_add_content_moderation,
        guardrail_chain_add_pii_detection,
        guardrail_chain_add_token_limit,
        create_telemetry,
    )

    chain_handle = create_middleware_chain()

    if logging:
        middleware_use_logging(chain_handle)

    if caching:
        ttl_ms = caching.get("ttl_ms", 60_000)
        middleware_use_caching(chain_handle, ttl_ms)

    guardrail_handle = None
    if guardrail:
        guardrail_handle = create_guardrail_chain()
        if "content_moderation" in guardrail:
            threshold = guardrail["content_moderation"].get("threshold", 0.7)
            guardrail_chain_add_content_moderation(guardrail_handle, threshold)
        if "pii_detection" in guardrail:
            action = guardrail["pii_detection"].get("action", "mask")
            guardrail_chain_add_pii_detection(guardrail_handle, action)
        if "token_limit" in guardrail:
            max_tokens = guardrail["token_limit"].get("max_tokens", 4096)
            guardrail_chain_add_token_limit(guardrail_handle, max_tokens)

    telemetry_handle = None
    if telemetry:
        telemetry_handle = create_telemetry()

    return MiddlewareConfig(
        logging=logging,
        caching=caching,
        guardrail=guardrail,
        telemetry=telemetry,
        _chain_handle=chain_handle,
        _guardrail_handle=guardrail_handle,
        _telemetry_handle=telemetry_handle,
    )
