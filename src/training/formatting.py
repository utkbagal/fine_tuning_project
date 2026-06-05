from __future__ import annotations

from typing import Any


def format_expected_output(output: dict[str, Any]) -> str:
    fields = [
        "Severity",
        "FileLine",
        "Category",
        "Issue",
        "WhyItMatters",
        "SuggestedFix",
        "PatchSnippet",
        "Confidence",
    ]
    lines = []
    for field in fields:
        value = output.get(field, "")
        lines.append(f"{field}: {value}")
    return "\n".join(lines)


def build_training_text(row: dict[str, Any]) -> str:
    task_instruction = str(row.get("task_instruction", "")).strip()
    inp = row.get("input", {})
    out = row.get("output", {})

    file_path = inp.get("file_path", "unknown")
    line_hint = inp.get("line_hint", "")
    context_summary = inp.get("context_summary", "")
    changed_code = inp.get("changed_code", "")

    prompt = (
        f"{task_instruction}\n\n"
        f"File: {file_path}:{line_hint}\n"
        f"Context: {context_summary}\n\n"
        f"Changed code:\n{changed_code}\n"
    )
    target = format_expected_output(out)
    return f"### Instruction\n{prompt}\n### Response\n{target}"
