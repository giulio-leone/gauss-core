"""
Eval — Evaluation runner with multiple scorers via Rust.
"""

from __future__ import annotations

import json
from typing import Any, Optional


class EvalRunner:
    """
    Evaluation runner backed by Rust.

    Example:
        >>> runner = EvalRunner(threshold=0.8)
        >>> runner.add_scorer("exact_match")
        >>> runner.add_scorer("contains")
    """

    def __init__(self, threshold: Optional[float] = None) -> None:
        from gauss._native import create_eval_runner

        self._handle = create_eval_runner(threshold)

    def add_scorer(self, scorer_type: str) -> "EvalRunner":
        """Add a scorer. Types: 'exact_match', 'contains', 'length_ratio'."""
        from gauss._native import eval_add_scorer

        eval_add_scorer(self._handle, scorer_type)
        return self

    def destroy(self) -> None:
        """Release the underlying Rust eval runner."""
        from gauss._native import destroy_eval_runner

        destroy_eval_runner(self._handle)

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass


def load_dataset_jsonl(jsonl: str) -> list[dict[str, Any]]:
    """Load evaluation dataset from JSONL string."""
    from gauss._native import load_dataset_jsonl as _load

    result = _load(jsonl)
    return json.loads(result) if isinstance(result, str) else result


def load_dataset_json(json_str: str) -> list[dict[str, Any]]:
    """Load evaluation dataset from JSON string."""
    from gauss._native import load_dataset_json as _load

    result = _load(json_str)
    return json.loads(result) if isinstance(result, str) else result
