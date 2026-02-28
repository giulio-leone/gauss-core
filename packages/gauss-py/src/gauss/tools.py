"""
Tools — Decorator-based tool definition for agents.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class ToolDef:
    """A tool definition with name, description, parameters, and execute function."""

    name: str
    description: str
    parameters: dict[str, Any] | None = None
    execute: Callable | None = None


def _infer_parameters(fn: Callable) -> dict[str, Any]:
    """Infer JSON Schema parameters from function signature."""
    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        annotation = param.annotation
        json_type = "string"

        if annotation != inspect.Parameter.empty:
            json_type = type_map.get(annotation, "string")

        prop: dict[str, Any] = {"type": json_type}
        properties[name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required

    return schema


def tool(
    description: str = "",
    name: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> Callable:
    """
    Decorator to define an agent tool.

    Example:
        >>> @tool(description="Get weather for a city")
        ... async def weather(city: str) -> str:
        ...     return f"Sunny in {city}"
    """

    def decorator(fn: Callable) -> Callable:
        tool_name = name or fn.__name__
        tool_desc = description or fn.__doc__ or ""
        tool_params = parameters or _infer_parameters(fn)

        tool_def = ToolDef(
            name=tool_name,
            description=tool_desc,
            parameters=tool_params,
            execute=fn,
        )

        fn._gauss_tool = tool_def  # type: ignore
        return fn

    return decorator
