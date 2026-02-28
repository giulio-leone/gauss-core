from __future__ import annotations

from dataclasses import dataclass

@dataclass
class Provider:
    handle: int
    provider_type: str
    model: str
    def destroy(self) -> None: ...

def gauss(
    provider_type: str,
    model: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    max_retries: int | None = None,
) -> Provider: ...
