"""Gradio UI: every dataset parameter is a component; Generate runs the pipeline."""

from __future__ import annotations

import csv
import os
import re
import tempfile
from typing import Any

import gradio as gr
from dotenv import load_dotenv

from .anthropic_client import AnthropicCompletor
from .generation.batches import generate_all_rows
from .generation.schema import request_schema
from .validation import assert_anthropic_env, validate_form


def _safe_filename_stem(name: str) -> str:
    """Turn dataset name into a short, filesystem-safe stem."""
    stem = re.sub(r"[^\w\-.]+", "_", name.strip(), flags=re.UNICODE)
    return (stem or "dataset")[:80]


def _write_csv(rows: list[dict[str, Any]], dataset_name: str) -> str:
    """Write UTF-8 CSV to a temp file; return path for ``gr.File``."""
    if not rows:
        raise ValueError("No rows to export.")

    fieldnames = list(rows[0].keys())
    stem = _safe_filename_stem(dataset_name)
    fd, path = tempfile.mkstemp(prefix=f"{stem}_", suffix=".csv", text=True)
    os.close(fd)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return path


def generate_clicked(
    dataset_name: str,
    description: str,
    num_rows: float,
    num_columns: float,
    required_features: str,
    claude_model: str,
    batch_rows: float | None,
) -> tuple[str | None, str]:
    """Validate Gradio inputs, run schema + batches via Anthropic, export CSV.

    Returns:
        ``(file_path_or_none, status_markdown)`` for ``gr.File`` and status text.
    """
    try:
        assert_anthropic_env()
        inputs = validate_form(
            dataset_name=dataset_name,
            description=description,
            num_rows=num_rows,
            num_columns=num_columns,
            required_features=required_features or "",
            claude_model=claude_model,
            batch_rows=batch_rows,
        )

        completer = AnthropicCompletor(inputs.claude_model)
        schema = request_schema(completer, inputs)
        rows = generate_all_rows(completer, inputs, schema)
        path = _write_csv(rows, inputs.dataset_name)

        status = (
            f"**Done.** Generated **{len(rows)}** rows and **{len(schema['columns'])}** columns.\n\n"
            f"- Rows per API call used: **{inputs.batch_rows}**\n"
            f"- File: `{path}`"
        )
        return path, status

    except ValueError as e:
        return None, f"**Validation error**\n\n{e}"
    except RuntimeError as e:
        return None, f"**Configuration error**\n\n{e}"
    except Exception as e:
        return None, f"**Generation failed** (`{type(e).__name__}`)\n\n{e}"


def build_ui() -> gr.Blocks:
    """Build ``gr.Blocks`` with Textbox/Number for dataset params, Generate, File output.

    Include:
        - Dataset name, description, # rows, # columns
        - Optional required-features text
        - Claude model (Textbox or Dropdown with editable value)
        - Optional number for rows per API call (nullable; validation applies default)
    """
    with gr.Blocks(title="Synthetic dataset generator") as demo:
        gr.Markdown(
            "# DatForge: Synthetic Dataset Generator\n\n"
            "Describe the dataset you want; Claude (Anthropic) proposes a **schema**, "
            "then fills **batches** of rows until the target row count is reached. "
            "Set **ANTHROPIC_API_KEY** in your environment or `.env`."
        )

        with gr.Row():
            dataset_name = gr.Textbox(
                label="Dataset name",
                placeholder="e.g. us_rental_listings",
            )
            claude_model = gr.Textbox(
                label="Claude model id",
                placeholder="e.g. claude-sonnet-4-20250514",
            )

        description = gr.Textbox(
            label="Dataset description",
            lines=5,
            placeholder="Domain, realism, correlations, edge cases…",
        )

        with gr.Row():
            num_rows = gr.Number(
                label="# of rows",
                minimum=1,
                precision=0,
                value=50,
            )
            num_columns = gr.Number(
                label="# of columns",
                minimum=1,
                precision=0,
                value=5,
            )
            batch_rows = gr.Number(
                label="Rows per API call (optional, default 50)",
                minimum=1,
                precision=0,
                value=None,
                info="Smaller batches are safer for JSON; larger = fewer API calls.",
            )

        required_features = gr.Textbox(
            label="Required features & value rules (optional)",
            lines=4,
            placeholder="e.g. house_type: single-family, apartment, duplex…",
        )

        generate_btn = gr.Button("Generate", variant="primary")

        status = gr.Markdown(value="Fill the form and click **Generate**.")
        csv_file = gr.File(label="Download CSV")

        inputs = [
            dataset_name,
            description,
            num_rows,
            num_columns,
            required_features,
            claude_model,
            batch_rows,
        ]
        generate_btn.click(
            fn=generate_clicked,
            inputs=inputs,
            outputs=[csv_file, status],
        )

    return demo


def main() -> None:
    """``load_dotenv()``, ``build_ui().launch()``."""
    load_dotenv()
    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    demo.launch()


if __name__ == "__main__":
    main()
