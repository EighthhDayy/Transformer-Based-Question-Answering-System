#!/usr/bin/env python3
"""Fast, dependency-free structural and syntax checks for the QA notebook.

This is NOT an end-to-end or GPU training test. Run the Colab notebook to
validate model downloads, optimizer compatibility, actual EM/F1 and QLoRA.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "Transformer_Based_Question_Answering_System.ipynb"


def normalize_notebook_magics(source: str) -> str:
    """Replace IPython shell commands/magics with parseable no-op lines."""
    result = []
    for line in source.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("%") or (stripped.startswith("!") and not stripped.startswith("!=")):
            indent = line[: len(line) - len(stripped)]
            result.append(indent + "pass  # notebook shell/magic command\n")
        else:
            result.append(line)
    return "".join(result)


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        raise ValueError("Expected Jupyter Notebook format version 4.")
    # GitHub's notebook renderer rejects incomplete ipywidgets state metadata.
    # This repository does not commit interactive widget state, only static
    # figures and archived text/HTML outputs.
    if "widgets" in notebook.get("metadata", {}):
        raise ValueError(
            "Remove metadata.widgets before commit; incomplete widget state "
            "causes GitHub's Invalid Notebook rendering error."
        )
    for cell_index, cell in enumerate(notebook["cells"]):
        for output in cell.get("outputs", []):
            if "application/vnd.jupyter.widget-view+json" in output.get("data", {}):
                raise ValueError(
                    f"Code cell {cell_index} contains widget-view output; "
                    "export a static figure or remove the stale progress widget."
                )
    cells = notebook["cells"]
    compiled = 0
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        if isinstance(source, list):
            source = "".join(source)
        try:
            ast.parse(
                normalize_notebook_magics(source),
                filename=f"{NOTEBOOK.name}:cell_{index}",
            )
        except SyntaxError as exc:
            raise SyntaxError(f"Code cell {index} failed: {exc}") from exc
        compiled += 1

    all_code = "\n".join(
        "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for cell in cells
        if cell["cell_type"] == "code"
    )
    required_features = {
        "extractive evaluation": 'evaluate.load("squad_v2")',
        "v5-compatible span decoder": "def extract_answer_span(",
        "fine-tuned QA checkpoint": "./squad_finetuned_model",
        "local generator checkpoint": "./gpt2_finetuned_squad",
        "LoRA fine-tuning": "LoraConfig(",
        "QLoRA opt-in": "RUN_QWEN_QLORA = False",
        "answer span preservation": "window_start_char <= span_start",
        "QLoRA completion-only loss": "completion_only_loss=True",
    }
    missing = [
        name for name, identifier in required_features.items()
        if identifier not in all_code
    ]
    if missing:
        raise AssertionError("Missing expected pipeline elements: " + ", ".join(missing))
    if 'pipeline("question-answering"' in all_code or "task=\"question-answering\"" in all_code:
        raise AssertionError("Removed Transformers v5 question-answering pipeline is still used.")
    print(f"PASS: Jupyter JSON, {compiled} Python code cells and key pipeline markers.")
    print("NOTE: GPU execution and performance claims still require a real Colab run.")


if __name__ == "__main__":
    main()
