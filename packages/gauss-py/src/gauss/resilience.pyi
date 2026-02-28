from __future__ import annotations

def create_fallback(provider_handles: list[int]) -> int: ...
def create_circuit_breaker(
    provider_handle: int,
    failure_threshold: int | None = None,
    recovery_timeout_ms: int | None = None,
) -> int: ...
def create_resilient(
    primary_handle: int,
    fallback_handles: list[int] | None = None,
    enable_circuit_breaker: bool = False,
) -> int: ...
