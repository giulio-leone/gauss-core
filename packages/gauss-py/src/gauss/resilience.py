"""
Resilience — Fallback providers, circuit breakers, and resilient providers via Rust.
"""

from __future__ import annotations

from typing import Optional


def create_fallback(provider_handles: list[int]) -> int:
    """Create a fallback provider from a list of provider handles. Returns a new provider handle."""
    from gauss._native import create_fallback_provider

    return create_fallback_provider(provider_handles)


def create_circuit_breaker(
    provider_handle: int,
    failure_threshold: Optional[int] = None,
    recovery_timeout_ms: Optional[int] = None,
) -> int:
    """Wrap a provider with a circuit breaker. Returns a new provider handle."""
    from gauss._native import create_circuit_breaker as _create_cb

    return _create_cb(provider_handle, failure_threshold, recovery_timeout_ms)


def create_resilient(
    primary_handle: int,
    fallback_handles: Optional[list[int]] = None,
    enable_circuit_breaker: bool = False,
) -> int:
    """Create a resilient provider with retry, fallback, and optional circuit breaker. Returns handle."""
    from gauss._native import create_resilient_provider

    return create_resilient_provider(
        primary_handle,
        fallback_handles or [],
        enable_circuit_breaker,
    )
