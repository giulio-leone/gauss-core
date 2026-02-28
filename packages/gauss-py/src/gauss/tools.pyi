from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict[str, Any] | None = None
    execute: Callable[..., Any] | None = None

def tool(
    description: str = "",
    name: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...
