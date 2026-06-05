from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any

from src.data.io_jsonl import read_jsonl, write_json


REQUIRED_TOP_LEVEL = {
    "id",
    "split",
    "language",
    "task_instruction",
    "input",
    "output",
    "metadata",
}

REQUIRED_OUTPUT = {
    "Severity",
    "FileLine",
    "Category",
    "Issue",
    "WhyItMatters",
    "SuggestedFix",
    "PatchSnippet",
    "Confidence",
}

FILE_LINE_RE = re.compile(r"^.+:[0-9]+(-[0-9]+)?$")


def _word_count(text: str) -> int:
    return len([w for w in text.split() if w])


def _validate_rows(rows: list[dict[str, Any]], expected_split: str) -> dict[str, Any]:
    issues: list[str] = []
    id_counter = Counter()
    category_counter = Counter()
    severity_counter = Counter()
    edge_cases = 0

    for idx, row in enumerate(rows, start=1):
        row_id = str(row.get("id", f"ROW_{idx}"))
        id_counter[row_id] += 1

        missing_top = sorted(REQUIRED_TOP_LEVEL - set(row.keys()))
        if missing_top:
            issues.append(f"{row_id}: missing top-level fields {missing_top}")
            continue

        if row["split"] != expected_split:
            issues.append(f"{row_id}: split mismatch ({row['split']} != {expected_split})")

        if not isinstance(row["output"], dict):
            issues.append(f"{row_id}: output must be an object")
            continue

        missing_output = sorted(REQUIRED_OUTPUT - set(row["output"].keys()))
        if missing_output:
            issues.append(f"{row_id}: missing output fields {missing_output}")
            continue

        out = row["output"]
        issue_text = str(out.get("Issue", "")).strip()
        if _word_count(issue_text) > 25:
            issues.append(f"{row_id}: Issue exceeds 25 words")

        file_line = str(out.get("FileLine", "")).strip()
        if not FILE_LINE_RE.match(file_line):
            issues.append(f"{row_id}: invalid FileLine '{file_line}'")

        confidence = out.get("Confidence")
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            issues.append(f"{row_id}: Confidence not numeric")
        else:
            if confidence_value < 0.0 or confidence_value > 1.0:
                issues.append(f"{row_id}: Confidence out of range [0,1]")

        if not str(out.get("SuggestedFix", "")).strip():
            issues.append(f"{row_id}: SuggestedFix is empty")

        if not str(out.get("WhyItMatters", "")).strip():
            issues.append(f"{row_id}: WhyItMatters is empty")

        category_counter[str(out.get("Category", "UNKNOWN"))] += 1
        severity_counter[str(out.get("Severity", "UNKNOWN"))] += 1

        edge_tag = str(row.get("input", {}).get("edge_case_tag", "none"))
        if edge_tag and edge_tag != "none":
            edge_cases += 1

    duplicate_ids = [k for k, v in id_counter.items() if v > 1]
    for dup in duplicate_ids:
        issues.append(f"{dup}: duplicate id")

    total = max(1, len(rows))
    return {
        "count": len(rows),
        "issues_count": len(issues),
        "issues_preview": issues[:30],
        "duplicate_ids_count": len(duplicate_ids),
        "category_distribution": dict(category_counter),
        "severity_distribution": dict(severity_counter),
        "edge_case_coverage_pct": round((edge_cases * 100.0) / total, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate capstone dataset JSONL files.")
    parser.add_argument("--train", required=True, type=Path, help="Path to train.jsonl")
    parser.add_argument("--eval", required=True, type=Path, help="Path to eval.jsonl")
    parser.add_argument("--report", required=True, type=Path, help="Output validation report JSON")
    args = parser.parse_args()

    train_rows = read_jsonl(args.train)
    eval_rows = read_jsonl(args.eval)

    report = {
        "train": _validate_rows(train_rows, "train"),
        "eval": _validate_rows(eval_rows, "eval"),
    }

    write_json(args.report, report)

    total_issues = report["train"]["issues_count"] + report["eval"]["issues_count"]
    print(f"Train rows: {report['train']['count']}")
    print(f"Eval rows: {report['eval']['count']}")
    print(f"Total issues: {total_issues}")
    print(f"Report written: {args.report}")


if __name__ == "__main__":
    main()
