from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from src.config.settings import load_settings
from src.data.io_jsonl import read_jsonl, write_json
from src.eval.metrics import approx_token_count, correctness_proxy, format_adherence_score
from src.inference.adapter_registry import resolve_latest_adapter
from src.inference.router import InferenceRouter


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_prompt(record: dict[str, Any]) -> str:
    instruction = str(record.get("task_instruction", "")).strip()
    inp = record.get("input", {})
    file_path = inp.get("file_path", "unknown")
    line_hint = inp.get("line_hint", "")
    context_summary = inp.get("context_summary", "")
    changed_code = inp.get("changed_code", "")

    return (
        f"{instruction}\n\n"
        f"File: {file_path}:{line_hint}\n"
        f"Context: {context_summary}\n\n"
        f"Changed code:\n{changed_code}\n"
    )


def evaluate(
    eval_rows: list[dict[str, Any]],
    sample_size: int | None = None,
    adapter_path: str | None = None,
) -> dict[str, Any]:
    rows = eval_rows if sample_size is None else eval_rows[:sample_size]
    router = InferenceRouter(adapter_path=adapter_path)

    per_variant_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)
    per_record_outputs: list[dict[str, Any]] = []

    for row in rows:
        print(f"Evaluating record id: {row.get('id', 'unknown')}...")
        prompt = build_prompt(row)
        results = router.run_all_with_timings(prompt)

        record_result = {
            "id": row.get("id"),
            "variant_outputs": [],
        }

        for result in results:
            variant = result["variant"]
            text = result.get("text", "")
            adherence = format_adherence_score(text)
            proxy = correctness_proxy(text, row)
            token_est = approx_token_count(text)
            latency_ms = float(result.get("latency_ms", 0.0))

            per_variant_metrics[variant].append(
                {
                    "format_adherence": adherence,
                    "category_match": proxy["category_match"],
                    "severity_match": proxy["severity_match"],
                    "lexical_overlap": proxy["lexical_overlap"],
                    "latency_ms": latency_ms,
                    "token_estimate": float(token_est),
                }
            )

            record_result["variant_outputs"].append(
                {
                    "variant": variant,
                    "latency_ms": latency_ms,
                    "token_estimate": token_est,
                    "format_adherence": adherence,
                    "category_match": proxy["category_match"],
                    "severity_match": proxy["severity_match"],
                    "lexical_overlap": proxy["lexical_overlap"],
                    "text_preview": text[:600],
                }
            )

        per_record_outputs.append(record_result)

    aggregates: dict[str, Any] = {}
    for variant, metrics in per_variant_metrics.items():
        count = max(1, len(metrics))
        aggregates[variant] = {
            "samples": len(metrics),
            "avg_format_adherence": round(sum(m["format_adherence"] for m in metrics) / count, 4),
            "avg_category_match": round(sum(m["category_match"] for m in metrics) / count, 4),
            "avg_severity_match": round(sum(m["severity_match"] for m in metrics) / count, 4),
            "avg_lexical_overlap": round(sum(m["lexical_overlap"] for m in metrics) / count, 4),
            "avg_latency_ms": round(sum(m["latency_ms"] for m in metrics) / count, 2),
            "avg_token_estimate": round(sum(m["token_estimate"] for m in metrics) / count, 2),
        }

    return {
        "generated_at": utc_now_iso(),
        "eval_rows": len(rows),
        "aggregates": aggregates,
        "records": per_record_outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 4-variant evaluation on eval dataset.")
    parser.add_argument("--eval", type=Path, default=None, help="Path to eval.jsonl")
    parser.add_argument("--report", type=Path, default=None, help="Output report JSON path")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional row limit for quick eval")
    parser.add_argument("--adapter-path", type=str, default=None, help="Optional adapter path override")
    parser.add_argument(
        "--golden-ids",
        type=Path,
        default=None,
        help="Optional path to golden_eval_ids.json. If provided, eval rows are filtered to listed ids.",
    )
    args = parser.parse_args()

    settings = load_settings()
    eval_path = args.eval or settings.eval_file
    report_path = args.report or (settings.reports_dir / "variant_eval_report.json")
    resolved_adapter = args.adapter_path or resolve_latest_adapter(settings.models_dir) or settings.adapter_path

    rows = read_jsonl(eval_path)
    golden_ids_path = args.golden_ids
    if golden_ids_path is not None:
        payload = json.loads(golden_ids_path.read_text(encoding="utf-8"))
        golden_ids = payload.get("ids", [])
        if not isinstance(golden_ids, list) or not all(isinstance(i, str) for i in golden_ids):
            raise ValueError("golden ids file must contain JSON object with string list field 'ids'.")
        golden_set = set(golden_ids)
        rows = [row for row in rows if str(row.get("id", "")) in golden_set]

    report = evaluate(rows, sample_size=args.sample_size, adapter_path=resolved_adapter)
    report["adapter_path_used"] = resolved_adapter
    if golden_ids_path is not None:
        report["golden_ids_path"] = str(golden_ids_path)

    write_json(report_path, report)
    print(f"Evaluated rows: {report['eval_rows']}")
    print(f"Adapter path used: {resolved_adapter}")
    print(f"Report written: {report_path}")


if __name__ == "__main__":
    main()
