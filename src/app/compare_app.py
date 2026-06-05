from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config.settings import load_settings
from src.data.io_jsonl import read_json, read_jsonl, write_json
from src.eval.run_eval import evaluate
from src.eval.metrics import approx_token_count, correctness_proxy, format_adherence_score
from src.inference.adapter_registry import resolve_latest_adapter
from src.inference.router import InferenceRouter


def _format_expected(record: dict) -> str:
    out = record.get("output", {})
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
    return "\n".join([f"{f}: {out.get(f, '')}" for f in fields])


def _build_prompt(record: dict) -> str:
    instruction = str(record.get("task_instruction", "")).strip()
    inp = record.get("input", {})
    return (
        f"{instruction}\n\n"
        f"File: {inp.get('file_path', 'unknown')}:{inp.get('line_hint', '')}\n"
        f"Context: {inp.get('context_summary', '')}\n\n"
        f"Changed code:\n{inp.get('changed_code', '')}\n"
    )


def _metric_row(result: dict, expected: dict | None = None) -> dict:
    text = result.get("text", "")
    row = {
        "variant": result.get("variant"),
        "format_adherence": format_adherence_score(text),
        "latency_ms": float(result.get("latency_ms", 0.0)),
        "token_estimate": approx_token_count(text),
    }
    if expected is not None:
        proxy = correctness_proxy(text, expected)
        row.update(proxy)
    return row


def _records_to_metric_df(records: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for rec in records:
        rec_id = rec.get("id")
        for v in rec.get("variant_outputs", []):
            rows.append(
                {
                    "id": rec_id,
                    "variant": v.get("variant"),
                    "format_adherence": float(v.get("format_adherence", 0.0)),
                    "latency_ms": float(v.get("latency_ms", 0.0)),
                    "token_estimate": float(v.get("token_estimate", 0.0)),
                    "category_match": float(v.get("category_match", 0.0)),
                    "severity_match": float(v.get("severity_match", 0.0)),
                    "lexical_overlap": float(v.get("lexical_overlap", 0.0)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    settings = load_settings()
    st.set_page_config(page_title="Capstone 4 Variant Compare", layout="wide")

    st.title("Capstone 4 - Base vs Fine-Tuned Comparison")

    latest_adapter = resolve_latest_adapter(settings.models_dir) or settings.adapter_path
    st.caption(f"Adapter path in use: {latest_adapter}")

    eval_path = settings.eval_file
    eval_rows: list[dict] = []
    if eval_path.exists():
        with eval_path.open("r", encoding="utf-8") as f:
            eval_rows = [json.loads(line) for line in f if line.strip()]

    default_prompt = "Review this code change and return structured review comments with Severity, Category, Issue, WhyItMatters, SuggestedFix, and Confidence."

    st.subheader("Golden Eval")
    golden_ids_path = settings.project_root / "data" / "golden" / "golden_eval_ids.json"
    golden_report_path = settings.reports_dir / "variant_eval_report_golden.json"
    golden_col_left, golden_col_right = st.columns([2, 1])
    with golden_col_left:
        st.caption(f"Golden ids file: {golden_ids_path}")
        st.caption(f"Golden report: {golden_report_path}")
    with golden_col_right:
        run_golden = st.button("Run Golden Eval", use_container_width=True)

    if run_golden:
        if not golden_ids_path.exists():
            st.error(f"Golden ids file not found: {golden_ids_path}")
        elif not settings.eval_file.exists():
            st.error(f"Eval file not found: {settings.eval_file}")
        else:
            payload = read_json(golden_ids_path)
            golden_ids = payload.get("ids", [])
            if not isinstance(golden_ids, list) or not all(isinstance(i, str) for i in golden_ids):
                st.error("Invalid golden ids file: expected object with string list field 'ids'.")
            else:
                golden_set = set(golden_ids)
                eval_rows_full = read_jsonl(settings.eval_file)
                eval_rows = [r for r in eval_rows_full if str(r.get("id", "")) in golden_set]
                report = evaluate(eval_rows, sample_size=None, adapter_path=latest_adapter)
                report["adapter_path_used"] = latest_adapter
                report["golden_ids_path"] = str(golden_ids_path)
                write_json(golden_report_path, report)

                st.success(f"Golden eval complete. Rows evaluated: {report.get('eval_rows', 0)}")
                agg = report.get("aggregates", {})
                if agg:
                    agg_rows = [{"variant": k, **v} for k, v in agg.items()]
                    agg_df = pd.DataFrame(agg_rows)
                    st.dataframe(agg_df, use_container_width=True)

                    fig_g1 = px.bar(
                        agg_df,
                        x="variant",
                        y="avg_format_adherence",
                        title="Golden Eval: Avg Format Adherence",
                        range_y=[0, 1],
                    )
                    fig_g2 = px.bar(agg_df, x="variant", y="avg_latency_ms", title="Golden Eval: Avg Latency (ms)")
                    fig_g3 = px.bar(
                        agg_df,
                        x="variant",
                        y="avg_token_estimate",
                        title="Golden Eval: Avg Token Estimate",
                    )
                    st.plotly_chart(fig_g1, use_container_width=True)
                    st.plotly_chart(fig_g2, use_container_width=True)
                    st.plotly_chart(fig_g3, use_container_width=True)

                    if {"avg_category_match", "avg_severity_match", "avg_lexical_overlap"}.issubset(agg_df.columns):
                        fig_g4 = px.bar(
                            agg_df,
                            x="variant",
                            y=["avg_category_match", "avg_severity_match", "avg_lexical_overlap"],
                            barmode="group",
                            title="Golden Eval: Avg Correctness Proxy Metrics",
                        )
                        st.plotly_chart(fig_g4, use_container_width=True)
                else:
                    st.warning("Golden eval finished but aggregates are empty.")

    if golden_report_path.exists():
        with st.expander("View Latest Golden Eval Report", expanded=False):
            st.json(read_json(golden_report_path))

    col_left, col_right = st.columns([2, 1])
    with col_left:
        prompt_mode = st.radio("Prompt source", ["Custom prompt", "From eval dataset"], horizontal=True)
        selected_row = None

        if prompt_mode == "From eval dataset" and eval_rows:
            max_idx = len(eval_rows) - 1
            row_idx = st.slider("Eval row index", min_value=0, max_value=max_idx, value=0)
            selected_row = eval_rows[row_idx]
            prompt_text = _build_prompt(selected_row)
            st.text_area("Prompt preview", value=prompt_text, height=260)
            with st.expander("Expected output (ground truth)", expanded=False):
                st.code(_format_expected(selected_row), language="text")
        else:
            prompt_text = st.text_area("Prompt", value=default_prompt, height=260)

    with col_right:
        run = st.button("Run 4-Variant Comparison", use_container_width=True)

    if run:
        router = InferenceRouter(adapter_path=latest_adapter)
        results = router.run_all_with_timings(prompt_text)

        metrics = [_metric_row(r, selected_row) for r in results]
        metric_df = pd.DataFrame(metrics)

        # ── Fixed-height 2x2 grid so all four boxes are equal-sized ──────────
        OUTPUT_BOX_HEIGHT = 420  # px — adjust to taste

        _card_css = f"""
        <style>
        .variant-card {{
            border: 1px solid #d0d0d0;
            border-radius: 8px;
            padding: 12px 14px;
            height: {OUTPUT_BOX_HEIGHT}px;
            overflow-y: auto;
            background: #fafafa;
            font-family: monospace;
            font-size: 0.82rem;
            white-space: pre-wrap;
            word-break: break-word;
        }}
        .variant-header {{
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 4px;
            color: #1a1a2e;
        }}
        .variant-meta {{
            font-size: 0.78rem;
            color: #666;
            margin-bottom: 6px;
        }}
        </style>
        """
        st.markdown(_card_css, unsafe_allow_html=True)

        st.subheader("Variant Outputs")
        top_row = st.columns(2)
        bot_row = st.columns(2)
        grid_cells = [top_row[0], top_row[1], bot_row[0], bot_row[1]]

        for cell, result in zip(grid_cells, results):
            variant = result.get("variant", "")
            text = result.get("text", "").replace("<", "&lt;").replace(">", "&gt;")
            latency = result.get("latency_ms", 0.0)
            meta = result.get("metadata") or {}
            meta_str = " | ".join(f"{k}: {v}" for k, v in meta.items()) if meta else ""
            with cell:
                st.markdown(
                    f"""
                    <div class="variant-header">{variant}</div>
                    <div class="variant-meta">latency: {latency} ms{"  &nbsp;·&nbsp;  " + meta_str if meta_str else ""}</div>
                    <div class="variant-card">{text}</div>
                    """,
                    unsafe_allow_html=True,
                )

        st.subheader("Metrics")
        st.dataframe(metric_df, use_container_width=True)

        fig1 = px.bar(metric_df, x="variant", y="format_adherence", title="Format Adherence by Variant", range_y=[0, 1])
        fig2 = px.bar(metric_df, x="variant", y="latency_ms", title="Latency by Variant (ms)")
        fig3 = px.bar(metric_df, x="variant", y="token_estimate", title="Token Estimate by Variant")

        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)

        if "category_match" in metric_df.columns and "severity_match" in metric_df.columns:
            fig4 = px.bar(
                metric_df,
                x="variant",
                y=["category_match", "severity_match", "lexical_overlap"],
                barmode="group",
                title="Correctness Proxy Metrics",
            )
            st.plotly_chart(fig4, use_container_width=True)


if __name__ == "__main__":
    main()
