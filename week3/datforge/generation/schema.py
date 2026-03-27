"""Phase A: one Anthropic call to produce and parse the column schema."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from ..validation import GenerationInputs
from .prompts import system_prompt_json_only, user_prompt_schema_phase

_ALLOWED_DTYPES = frozenset(
    {"string", "integer", "float", "categorical", "boolean", "date"}
)
_SNAKE_CASE = re.compile(r"^[a-z][a-z0-9_]*$")


class _TextCompleter(Protocol):
    """Minimal interface for the Anthropic wrapper (or tests)."""

    def complete(self, system_prompt: str, user_prompt: str) -> str: ...


def _strip_markdown_fence(text: str) -> str:
    """Remove leading/trailing Markdown code fences from a model response.

    Models sometimes wrap JSON in fenced blocks (e.g. lines starting with
    triple backticks) despite instructions not to. This keeps ``json.loads``
    viable without failing on the fence lines.

    Args:
        text: Raw assistant text, optionally starting with a fence line.

    Returns:
        Inner content with outer fences removed; unchanged if no fence starts
        the string after strip.
    """
    t = text.strip()
    if not t.startswith("```"):
        return t
    lines = t.split("\n")
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_json_object(raw: str) -> dict[str, Any]:
    """Parse one JSON object from assistant text.

    Strips optional fences, then parses the whole string. If that fails,
    extracts the substring from the first ``{`` to the last ``}`` and parses
    that (handles trailing prose or minor junk outside the object).

    Args:
        raw: Full model response string.

    Returns:
        The parsed top-level JSON object.

    Raises:
        ValueError: If no object can be parsed or the top-level value is not
            a JSON object.
    """
    s = _strip_markdown_fence(raw.strip())
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(
                "Model response is not valid JSON (could not find an object)."
            ) from None
        try:
            data = json.loads(s[start : end + 1])
        except json.JSONDecodeError as e:
            raise ValueError(f"Model response is not valid JSON: {e}") from e

    if not isinstance(data, dict):
        raise ValueError("Parsed JSON must be a single object at the top level.")
    return data


def _validate_column_entry(obj: Any, index: int) -> dict[str, Any]:
    """Validate and normalize a single element of the ``columns`` array.

    Enforces required keys ``name``, ``dtype``, and ``description``, optional
    ``allowed_values``, ``min``, and ``max``, and allowed dtype / naming rules.

    Args:
        obj: One column entry from parsed JSON (expected to be a dict).
        index: Zero-based index in ``columns`` (for error messages).

    Returns:
        A new dict with stripped strings and only defined optional keys copied.

    Raises:
        ValueError: If the entry is malformed or violates schema rules.
    """
    if not isinstance(obj, dict):
        raise ValueError(f"columns[{index}] must be an object.")

    name = obj.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f'columns[{index}].name must be a non-empty string.')
    name = name.strip()
    if not _SNAKE_CASE.match(name):
        raise ValueError(
            f'columns[{index}].name must be snake_case (got {name!r}).'
        )

    dtype = obj.get("dtype")
    if not isinstance(dtype, str) or dtype not in _ALLOWED_DTYPES:
        raise ValueError(
            f'columns[{index}].dtype must be one of {sorted(_ALLOWED_DTYPES)} '
            f"(got {dtype!r})."
        )

    desc = obj.get("description")
    if not isinstance(desc, str):
        raise ValueError(f'columns[{index}].description must be a string.')
    desc = desc.strip()
    if not desc:
        raise ValueError(f'columns[{index}].description must be non-empty.')

    out: dict[str, Any] = {
        "name": name,
        "dtype": dtype,
        "description": desc,
    }

    if "allowed_values" in obj and obj["allowed_values"] is not None:
        av = obj["allowed_values"]
        if not isinstance(av, list) or not all(isinstance(x, str) for x in av):
            raise ValueError(
                f"columns[{index}].allowed_values must be an array of strings or omitted."
            )
        out["allowed_values"] = list(av)

    for key in ("min", "max"):
        if key in obj and obj[key] is not None:
            v = obj[key]
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ValueError(
                    f"columns[{index}].{key} must be a number or omitted."
                )
            out[key] = v

    return out


def _validate_schema_payload(data: dict[str, Any], expected_columns: int) -> dict[str, Any]:
    """Validate the full schema object after JSON parsing.

    Ensures ``data`` has a ``columns`` list of length ``expected_columns``,
    validates each column via ``_validate_column_entry``, and checks for
    duplicate column names.

    Args:
        data: Top-level object from the model (must include ``columns``).
        expected_columns: Required length of ``columns`` (from user input).

    Returns:
        A dict ``{"columns": [...]}`` with normalized column entries.

    Raises:
        ValueError: If structure, length, or any column entry is invalid.
    """
    cols = data.get("columns")
    if not isinstance(cols, list):
        raise ValueError('Schema must contain a "columns" array.')
    if len(cols) != expected_columns:
        raise ValueError(
            f'Schema must contain exactly {expected_columns} columns '
            f"(got {len(cols)})."
        )

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, item in enumerate(cols):
        col = _validate_column_entry(item, i)
        if col["name"] in seen:
            raise ValueError(f'Duplicate column name: {col["name"]!r}.')
        seen.add(col["name"])
        normalized.append(col)

    return {"columns": normalized}


def request_schema(completer: _TextCompleter, inputs: GenerationInputs) -> dict[str, Any]:
    """Call Claude with schema-phase prompts and parse the returned JSON object.

    Expects the shape produced by ``user_prompt_schema_phase``:
    ``{ "columns": [ { "name", "dtype", "description", ... }, ... ] }``.

    Raises:
        ValueError: If the response is not valid JSON or misses expected structure.

    Returns:
        A dict with key ``columns`` listing validated column specs for Phase B.
    """
    raw = completer.complete(
        system_prompt_json_only(),
        user_prompt_schema_phase(inputs),
    )
    data = _parse_json_object(raw)
    return _validate_schema_payload(data, inputs.num_columns)
