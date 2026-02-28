from __future__ import annotations

from dataclasses import dataclass
from typing import Any

@dataclass
class MiddlewareConfig:
    guardrail_handle: int | None = None
    telemetry_handle: int | None = None
    resilient_provider_handle: int | None = None

def middleware(
    guardrail: dict[str, Any] | None = None,
    telemetry: bool = False,
    resilience: dict[str, Any] | None = None,
) -> MiddlewareConfig: ...
