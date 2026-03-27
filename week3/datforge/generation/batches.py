"""Phase B: repeated Anthropic calls until ``inputs.num_rows`` rows exist."""

from __future__ import annotations

import json
from typing import Any, Protocol

from ..validation import GenerationInputs
from .prompts import system_prompt_json_only, user_prompt_row_batch


class _TextCompleter(Protocol):
    """Minimal interface for the Anthropic wrapper (or tests)."""

    def complete(self, system_prompt: str, user_prompt: str) -> str: 
        ...


def _strip_markdown_fence(text: str) -> str:
    """Remove leading/trailing Markdown code fences (same idea as ``schema`` module)."""
    t = text.strip()
    if not t.startswith("```"):
        return t
    lines = t.split("\n")
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_json_array(raw: str) -> list[Any]:
    """Parse a JSON array from assistant text, tolerating fences and trailing junk."""
    s = _strip_markdown_fence(raw.strip())
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        start = s.find("[")
        end = s.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(
                "Model response is not valid JSON (could not find an array)."
            ) from None
        try:
            data = json.loads(s[start : end + 1])
        except json.JSONDecodeError as e:
            raise ValueError(f"Model response is not valid JSON: {e}") from e

    if not isinstance(data, list):
        raise ValueError("Parsed JSON must be a top-level array of row objects.")
    return data


def _column_names(schema: dict[str, Any]) -> list[str]:
    """Ordered CSV column names from a validated schema dict."""
    cols = schema.get("columns")
    if not isinstance(cols, list):
        raise ValueError('schema must contain a "columns" list.')
    names: list[str] = []
    for i, c in enumerate(cols):
        if not isinstance(c, dict):
            raise ValueError(f"schema.columns[{i}] must be an object.")
        n = c.get("name")
        if not isinstance(n, str) or not n.strip():
            raise ValueError(f"schema.columns[{i}].name must be a non-empty string.")
        names.append(n.strip())
    return names


def _validate_batch_rows(
    rows: list[Any],
    column_names: list[str],
    expected_len: int,
) -> list[dict[str, Any]]:
    """Ensure batch length and each row matches the schema column set."""
    if len(rows) > expected_len:
        rows = rows[:expected_len]
    if len(rows) != expected_len:
        raise ValueError(
            f"Expected exactly {expected_len} rows in batch, got {len(rows)}."
        )

    expected = set(column_names)
    normalized: list[dict[str, Any]] = []

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Batch row {i} must be a JSON object.")
        keys = set(row.keys())
        if keys != expected:
            missing = expected - keys
            extra = keys - expected
            msg = f"Batch row {i} keys do not match schema."
            if missing:
                msg += f" Missing: {sorted(missing)}."
            if extra:
                msg += f" Extra: {sorted(extra)}."
            raise ValueError(msg)
        normalized.append({k: row[k] for k in column_names})

    return normalized


_MAX_BATCH_ATTEMPTS = 5


def _complete_batch(
    completer: _TextCompleter,
    inputs: GenerationInputs,
    schema: dict[str, Any],
    batch_index: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    """One API call; parse and validate rows. Retries with corrective hints on failure."""
    column_names = _column_names(schema)
    last_err: ValueError | None = None
    base_user = user_prompt_row_batch(inputs, schema, batch_index, batch_size)

    for attempt in range(_MAX_BATCH_ATTEMPTS):
        repair = ""
        if attempt > 0 and last_err is not None:
            repair = (
                "\n\n### Correction required\n"
                f"The previous answer was rejected: {last_err}\n"
                f"Return a JSON array with **exactly {batch_size}** objects—no more, no fewer. "
                "JSON only, no markdown fences."
            )
        raw = completer.complete(
            system_prompt_json_only(),
            base_user + repair,
        )
        try:
            parsed = _parse_json_array(raw)
            return _validate_batch_rows(parsed, column_names, batch_size)
        except ValueError as e:
            last_err = e

    assert last_err is not None
    raise last_err


def generate_all_rows(
    completer: _TextCompleter,
    inputs: GenerationInputs,
    schema: dict[str, Any],
) -> list[dict[str, Any]]:
    """Loop batch requests using ``inputs.batch_rows`` until enough rows are collected.

    Each batch is validated for length and for column keys matching ``schema``.
    Row order is stable: columns follow the order in ``schema["columns"]``.

    Args:
        completer: Backend implementing ``complete(system_prompt, user_prompt)``.
        inputs: Validated Gradio parameters (includes ``num_rows`` and ``batch_rows``).
        schema: Validated schema dict from ``request_schema`` (must include ``columns``).

    Returns:
        List of row dicts suitable for CSV export (length ``inputs.num_rows``).

    Raises:
        ValueError: If repeated batch responses fail validation.
    """
    if inputs.num_rows < 1:
        raise ValueError("num_rows must be at least 1.")

    all_rows: list[dict[str, Any]] = []
    batch_index = 0

    while len(all_rows) < inputs.num_rows:
        need = inputs.num_rows - len(all_rows)
        batch_size = min(inputs.batch_rows, need)
        batch = _complete_batch(
            completer,
            inputs,
            schema,
            batch_index=batch_index,
            batch_size=batch_size,
        )
        all_rows.extend(batch)
        batch_index += 1

    return all_rows
