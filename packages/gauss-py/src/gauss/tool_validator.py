"""
ToolValidator — Input validation and coercion via Rust patterns module.
"""

from __future__ import annotations

import json
from typing import Any, Optional


class ToolValidator:
    """
    Tool input validator backed by Rust.

    Example:
        >>> validator = ToolValidator(strategies=["type_cast", "null_to_default"])
        >>> result = validator.validate({"count": "5"}, {"type": "object", "properties": {"count": {"type": "integer"}}})
    """

    def __init__(self, strategies: Optional[list[str]] = None) -> None:
        from gauss._native import create_tool_validator

        self._handle = create_tool_validator(strategies)

    def validate(self, input_data: Any, schema: dict[str, Any]) -> Any:
        """Validate and optionally coerce input against a JSON schema."""
        from gauss._native import tool_validator_validate

        result_json = tool_validator_validate(
            self._handle,
            json.dumps(input_data),
            json.dumps(schema),
        )
        return json.loads(result_json)

    def destroy(self) -> None:
        """Release the underlying Rust tool validator."""
        from gauss._native import destroy_tool_validator

        destroy_tool_validator(self._handle)

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass
