"""Validate Gradio field values before calling Anthropic or writing files."""

import os
from dataclasses import dataclass
from typing import Optional

# Upper bounds for accidental huge requests (validation only—not user dataset "config").
_MAX_ROWS = 10_000
_MAX_COLUMNS = 100
_DEFAULT_BATCH_ROWS = 50
_MIN_BATCH_ROWS = 1
_MAX_BATCH_ROWS = 500


def _as_whole_number_in_range(
    label: str,
    value: float | int | None,
    lo: int,
    hi: int,
) -> int:
    """Parse Gradio numeric input as an integer within ``[lo, hi]``."""
    if value is None:
        raise ValueError(f"{label} is required.")
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number.") from None
    if not x.is_integer():
        raise ValueError(f"{label} must be a whole number.")
    n = int(x)
    if n < lo or n > hi:
        raise ValueError(f"{label} must be between {lo} and {hi} (inclusive).")
    return n

@dataclass(frozen=True)
class GenerationInputs:
    """All dataset generation parameters supplied by the user through Gradio.

    Attributes:
        dataset_name: Non-empty label; used in prompts and output filename stem.
        description: Non-empty description of domain and desired data behavior.
        num_rows: Positive integer; target number of CSV rows.
        num_columns: Positive integer; target number of columns in the schema.
        required_features: Optional free text listing required columns and value domains.
        claude_model: Anthropic model id (e.g. claude-sonnet-4-*); chosen by the user.
        batch_rows: Rows to request per API call in Phase B; must be >= 1.
    """

    dataset_name: str
    description: str
    num_rows: int
    num_columns: int
    required_features: Optional[str]
    claude_model: str
    batch_rows: int


def validate_form(
    dataset_name: str,
    description: str,
    num_rows: float | int,
    num_columns: float | int,
    required_features: str,
    claude_model: str,
    batch_rows: float | int | None,
) -> GenerationInputs:
    """Convert raw Gradio values into ``GenerationInputs`` with checks.

    Apply reasonable upper bounds inline (e.g. cap max rows/columns) to avoid
    accidental huge requests—those limits are validation rules here, not a
    separate config module.

    Raises:
        ValueError: If mandatory fields are empty or numeric inputs are invalid.

    Returns:
        A frozen ``GenerationInputs`` instance for the generation pipeline.
    """
    name = (dataset_name or "").strip()
    if not name:
        raise ValueError("Dataset name is required.")

    desc = (description or "").strip()
    if not desc:
        raise ValueError("Dataset description is required.")

    n_rows = _as_whole_number_in_range("# of rows", num_rows, 1, _MAX_ROWS)
    n_cols = _as_whole_number_in_range("# of columns", num_columns, 1, _MAX_COLUMNS)

    feats = (required_features or "").strip()
    required_features_out: Optional[str] = feats if feats else None

    model = (claude_model or "").strip()
    if not model:
        raise ValueError("Claude model is required (e.g. claude-sonnet-4-20250514).")

    if batch_rows is None:
        batch = _DEFAULT_BATCH_ROWS
    else:
        batch = _as_whole_number_in_range(
            "Rows per API call",
            batch_rows,
            _MIN_BATCH_ROWS,
            _MAX_BATCH_ROWS,
        )
    if batch > n_rows:
        batch = n_rows

    return GenerationInputs(
        dataset_name=name,
        description=desc,
        num_rows=n_rows,
        num_columns=n_cols,
        required_features=required_features_out,
        claude_model=model,
        batch_rows=batch,
    )


def assert_anthropic_env() -> None:
    """Ensure ``ANTHROPIC_API_KEY`` is set in the environment.

    Raises:
        RuntimeError: With a clear, user-facing message if the key is missing.
    """
    key = os.getenv("ANTHROPIC_API_KEY")
    if key is None or not str(key).strip():
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set or is empty. Add it to your environment "
            "or `.env` file (and use python-dotenv to load it before calling the API)."
        )
