"""Build prompts from ``GenerationInputs`` and intermediate JSON artifacts."""

from __future__ import annotations

import json
from typing import Any

from ..validation import GenerationInputs


def system_prompt_json_only() -> str:
    """Return the system message that instructs Claude to emit parseable JSON only."""
    return (
        "You generate synthetic tabular data for machine learning and analytics. "
        "Your answers are consumed by automated parsers.\n\n"
        "Strict output rules:\n"
        "1. Output a single JSON value only—valid UTF-8 JSON with double-quoted keys "
        "and strings.\n"
        "2. Do not wrap the JSON in markdown code fences or backticks.\n"
        "3. Do not add explanations, labels, or any text before or after the JSON.\n"
        "4. Escape characters inside strings per JSON rules.\n"
        "5. Honor the user's column count, names, types, and value constraints exactly."
    )


def user_prompt_schema_phase(inputs: GenerationInputs) -> str:
    """
    Ask for a JSON object describing columns: names, types, and allowed values.

    Must include ``inputs.required_features`` when non-empty.
    """
    lines = [
        "Design a column schema for a synthetic dataset that will be exported as CSV.",
        "",
        f"Dataset name: {inputs.dataset_name}",
        f"Dataset description: {inputs.description}",
        f"Required number of columns: {inputs.num_columns}",
        "",
        "Return one JSON object with this exact top-level shape:",
        '{ "columns": [ /* exactly '
        + str(inputs.num_columns)
        + " column objects */ ] }",
        "",
        "Each element of `columns` must be an object with:",
        '  - "name": string, snake_case, unique, suitable as a CSV header',
        '  - "dtype": one of "string", "integer", "float", "categorical", "boolean", "date"',
        '  - "description": short plain text describing the variable',
        '  - "allowed_values": optional array of strings; use for categorical columns '
        "when values are enumerated",
        '  - "min": optional number (for numeric ranges, inclusive lower bound)',
        '  - "max": optional number (for numeric ranges, inclusive upper bound)',
        "",
        "The `columns` array length must equal the required number of columns.",
        "Choose dtypes and descriptions that fit the dataset description and are "
        "realistic together (plausible correlations implied by the domain).",
    ]

    if inputs.required_features:
        lines.extend(
            [
                "",
                "Required columns / value rules from the user (must be reflected in "
                "the schema; the `columns` array must still have exactly the required "
                "length):",
                "",
                inputs.required_features.strip(),
            ]
        )

    lines.extend(
        [
            "",
            "Respond with the JSON object only.",
        ]
    )

    return "\n".join(lines)


def user_prompt_row_batch(
    inputs: GenerationInputs,
    schema: dict[str, Any],
    batch_index: int,
    batch_size: int,
) -> str:
    """
    Ask for a JSON array of length ``batch_size`` matching ``schema``.

    Each element must be one row object keyed by column names from the schema.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if batch_index < 0:
        raise ValueError("batch_index must be non-negative.")

    schema_json = json.dumps(schema, ensure_ascii=False, indent=2)

    return "\n".join(
        [
            "Generate synthetic data rows for the following dataset.",
            "",
            f"Dataset name: {inputs.dataset_name}",
            f"Dataset description: {inputs.description}",
            f"Batch index (0-based): {batch_index} — vary values across batches so "
            "rows are not duplicates of prior batches.",
            "",
            "Column schema (JSON). Every row object must use these column names and "
            "respect dtypes, allowed_values, and min/max where given:",
            schema_json,
            "",
            f"Return a JSON array containing exactly {batch_size} objects.",
            "Each object is one row: keys are the column `name` values from the schema; "
            "values are JSON primitives or strings (dates as ISO 8601 date strings "
            "YYYY-MM-DD unless the schema implies otherwise).",
            "",
            "Rules:",
            "- No missing keys: every row must include every column from the schema.",
            "- For categorical columns, only use values from allowed_values when present.",
            "- For numeric columns, stay within min/max when present.",
            "- Keep values coherent with the dataset description.",
            "",
            "Output the JSON array only.",
        ]
    )
