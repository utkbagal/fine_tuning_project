from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config.settings import load_settings
from src.data.io_jsonl import read_json, read_jsonl, write_json
from src.eval.run_eval import evaluate
from src.eval.metrics import approx_token_count, correctness_proxy, format_adherence_score
from src.inference.adapter_registry import resolve_latest_adapter
from src.inference.router import InferenceRouter

# ── Per-variant brand colours ────────────────────────────────────────────────
VARIANT_COLORS: dict[str, str] = {
    "base":               "#6366f1",   # indigo
    "base_plus_rag":      "#06b6d4",   # cyan
    "fine_tuned":         "#f59e0b",   # amber
    "fine_tuned_plus_rag": "#10b981",  # emerald
}
VARIANT_LABELS: dict[str, str] = {
    "base":               "Base",
    "base_plus_rag":      "Base + RAG",
    "fine_tuned":         "Fine-Tuned",
    "fine_tuned_plus_rag": "Fine-Tuned + RAG",
}

_GLOBAL_CSS = """
<style>
/* ── page chrome ── */
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"] { background: #161b27; }
h1, h2, h3 { color: #f1f5f9 !important; letter-spacing: -0.02em; }
p, label, .stCaption, .stMarkdown { color: #94a3b8 !important; }

/* ── pill badge ── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #0f1117;
    margin-right: 4px;
}

/* ── model info card ── */
.info-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 16px;
}
.info-card .label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 2px;
}
.info-card .value {
    font-size: 0.92rem;
    color: #e2e8f0;
    font-weight: 500;
    word-break: break-all;
}
.param-pill {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.78rem;
    color: #94a3b8;
    margin: 3px 3px 3px 0;
}
.param-label {
    color: #64748b;
    font-weight: 600;
}

/* ── variant output cards ── */
.vcard-wrap { border-radius: 12px; overflow: hidden; margin-bottom: 4px; }
.vcard-header {
    padding: 10px 14px 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.vcard-header .vname {
    font-weight: 700;
    font-size: 0.88rem;
    color: #f1f5f9;
}
.vcard-header .vmeta {
    font-size: 0.75rem;
    color: #64748b;
    margin-left: auto;
}
.vcard-body {
    background: #1e293b;
    padding: 12px 14px;
    height: 380px;
    overflow-y: auto;
    font-family: ui-monospace, Menlo, Consolas, monospace;
    font-size: 0.78rem;
    color: #cbd5e1;
    white-space: pre-wrap;
    word-break: break-word;
    border-top: 1px solid #0f172a;
}

/* ── metric chip ── */
.metric-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 8px 0; }
.mchip {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 8px 14px;
    text-align: center;
    min-width: 80px;
}
.mchip .mv { font-size: 1.1rem; font-weight: 700; color: #f1f5f9; }
.mchip .mk { font-size: 0.65rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; }

/* ── section divider ── */
.sec-divider {
    border: none;
    border-top: 1px solid #1e293b;
    margin: 24px 0;
}
</style>
"""

_PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", family="Inter, sans-serif", size=12),
    xaxis=dict(gridcolor="#1e293b", linecolor="#334155", tickcolor="#475569"),
    yaxis=dict(gridcolor="#1e293b", linecolor="#334155", tickcolor="#475569"),
    margin=dict(l=16, r=16, t=40, b=16),
)


def _color(variant: str) -> str:
    return VARIANT_COLORS.get(variant, "#94a3b8")


def _label(variant: str) -> str:
    return VARIANT_LABELS.get(variant, variant)


def _badge_html(variant: str) -> str:
    color = _color(variant)
    label = _label(variant)
    return f'<span class="badge" style="background:{color}">{label}</span>'


def _format_expected(record: dict) -> str:
    out = record.get("output", {})
    fields = ["Severity", "FileLine", "Category", "Issue",
              "WhyItMatters", "SuggestedFix", "PatchSnippet", "Confidence"]
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
        row.update(correctness_proxy(text, expected))
    return row


def _colored_bar(df: pd.DataFrame, y: str, title: str, range_y: list | None = None) -> go.Figure:
    variants = df["variant"].tolist()
    colors = [_color(v) for v in variants]
    labels = [_label(v) for v in variants]
    fig = go.Figure(go.Bar(
        x=labels, y=df[y].tolist(),
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:.3f}" if isinstance(v, float) else str(v) for v in df[y].tolist()],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
    ))
    fig.update_layout(title=dict(text=title, font=dict(color="#e2e8f0", size=14)),
                      **_PLOTLY_THEME)
    if range_y:
        fig.update_yaxes(range=range_y)
    return fig


def _radar_chart(df: pd.DataFrame) -> go.Figure:
    metrics = ["format_adherence", "category_match", "severity_match", "lexical_overlap"]
    labels = ["Format", "Category", "Severity", "Lexical"]
    fig = go.Figure()
    for _, row in df.iterrows():
        v = row.get("variant", "")
        values = [float(row.get(m, 0.0)) for m in metrics] + [float(row.get(metrics[0], 0.0))]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels + [labels[0]],
            fill="toself",
            name=_label(v),
            line=dict(color=_color(v), width=2),
            fillcolor=_color(v).replace(")", ",0.15)").replace("rgb", "rgba") if "rgb" in _color(v) else _color(v) + "26",
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="#1e293b",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#334155",
                            tickcolor="#475569", color="#64748b"),
            angularaxis=dict(gridcolor="#334155", color="#64748b"),
        ),
        showlegend=True,
        legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
        title=dict(text="Quality Radar", font=dict(color="#e2e8f0", size=14)),
        **{k: v for k, v in _PLOTLY_THEME.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def _load_adapter_manifest(adapter_path: str) -> dict:
    p = Path(adapter_path) / "adapter_manifest.json"
    if p.exists():
        try:
            return read_json(p)
        except Exception:
            pass
    return {}


def _list_saved_reports(reports_dir: Path) -> list[Path]:
    # Include full, generic, and golden variant reports; newest first.
    patterns = [
        "variant_eval_report*.json",
    ]
    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for p in reports_dir.glob(pattern):
            if p.is_file() and p not in seen:
                files.append(p)
                seen.add(p)
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return files


def _render_previous_run_panel(settings) -> None:
    st.markdown('<hr class="sec-divider">', unsafe_allow_html=True)
    st.markdown("### Previous Comparison Runs")

    report_files = _list_saved_reports(settings.reports_dir)
    if not report_files:
        st.info("No saved comparison reports found yet. Run an eval once to populate this list.")
        return

    label_to_path: dict[str, Path] = {}
    for p in report_files:
        ts = p.stat().st_mtime
        label = f"{p.name}  ({pd.to_datetime(ts, unit='s').strftime('%Y-%m-%d %H:%M:%S')})"
        label_to_path[label] = p

    col_sel, col_btn = st.columns([4, 1])
    with col_sel:
        selected_label = st.selectbox(
            "Select saved report",
            options=list(label_to_path.keys()),
            key="previous_report_select",
        )
    with col_btn:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        load_previous = st.button("Load Report", use_container_width=True, key="load_previous_report")

    if load_previous:
        selected_path = label_to_path[selected_label]
        try:
            report = read_json(selected_path)
        except Exception as exc:
            st.error(f"Failed to read report: {selected_path} ({exc})")
            return

        if not isinstance(report, dict) or "aggregates" not in report:
            st.error("Selected file does not look like a valid comparison report (missing 'aggregates').")
            return

        st.success(f"Loaded: {selected_path.name}")
        _render_golden_results(report)
        with st.expander("View loaded report JSON", expanded=False):
            st.json(report)


def _render_model_info(settings, adapter_path: str) -> None:
    manifest = _load_adapter_manifest(adapter_path)
    hp = manifest.get("hyperparameters", {})
    adapter_ready = (Path(adapter_path) / "adapter_config.json").exists()

    st.markdown(
        f"""
        <div class="info-card">
            <div class="label">Base Model</div>
            <div class="value">{settings.model_source or settings.base_model_id}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    adapter_status_color = "#10b981" if adapter_ready else "#f59e0b"
    adapter_status_text = "Ready" if adapter_ready else "Pending (dry-run placeholder)"
    st.markdown(
        f"""
        <div class="info-card">
            <div class="label">LoRA Adapter</div>
            <div class="value" style="font-size:0.8rem;color:#64748b;margin-bottom:6px">{adapter_path}</div>
            <span class="badge" style="background:{adapter_status_color}">{adapter_status_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if hp:
        pills = "".join([
            f'<span class="param-pill"><span class="param-label">rank</span> {hp.get("rank","—")}</span>',
            f'<span class="param-pill"><span class="param-label">alpha</span> {hp.get("alpha","—")}</span>',
            f'<span class="param-pill"><span class="param-label">lr</span> {hp.get("learning_rate","—")}</span>',
            f'<span class="param-pill"><span class="param-label">epochs</span> {hp.get("epochs","—")}</span>',
            f'<span class="param-pill"><span class="param-label">batch</span> {hp.get("batch_size","—")}</span>',
            f'<span class="param-pill"><span class="param-label">grad_accum</span> {hp.get("gradient_accumulation_steps","—")}</span>',
        ] + [f'<span class="param-pill"><span class="param-label">target</span> {m}</span>'
             for m in hp.get("target_modules", [])])
        st.markdown(
            f'<div class="info-card"><div class="label">Fine-Tuning Hyperparameters</div>'
            f'<div style="margin-top:8px">{pills}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="info-card"><div class="label">Fine-Tuning Hyperparameters</div>'
            '<div class="value" style="color:#475569;font-style:italic">Not available — run training to populate.</div></div>',
            unsafe_allow_html=True,
        )


def _render_variant_grid(results: list[dict]) -> None:
    top, bot = st.columns(2), st.columns(2)
    cells = [top[0], top[1], bot[0], bot[1]]
    for cell, result in zip(cells, results):
        v = result.get("variant", "")
        color = _color(v)
        text = result.get("text", "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        latency = result.get("latency_ms", 0.0)
        meta = result.get("metadata") or {}
        meta_str = " · ".join(f"{k}: {v2}" for k, v2 in meta.items()) if meta else ""
        meta_display = f"⏱ {latency} ms" + (f"  ·  {meta_str}" if meta_str else "")
        with cell:
            st.markdown(
                f"""
                <div class="vcard-wrap" style="border:1px solid {color}33">
                  <div class="vcard-header" style="background:{color}18;border-bottom:1px solid {color}33">
                    <span style="width:10px;height:10px;border-radius:50%;background:{color};display:inline-block;flex-shrink:0"></span>
                    <span class="vname">{_label(v)}</span>
                    <span class="vmeta">{meta_display}</span>
                  </div>
                  <div class="vcard-body">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_golden_panel(settings, latest_adapter: str) -> None:
    golden_ids_path = settings.project_root / "data" / "golden" / "golden_eval_ids.json"
    golden_report_path = settings.reports_dir / "variant_eval_report_golden.json"

    st.markdown('<hr class="sec-divider">', unsafe_allow_html=True)
    st.markdown("### 🏆 Golden Eval")

    g_left, g_right = st.columns([3, 1])
    with g_left:
        st.caption(f"IDs file: `{golden_ids_path}`")
        st.caption(f"Report:  `{golden_report_path}`")
    with g_right:
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
                all_rows = read_jsonl(settings.eval_file)
                filtered = [r for r in all_rows if str(r.get("id", "")) in golden_set]
                with st.spinner(f"Evaluating {len(filtered)} golden rows across 4 variants…"):
                    report = evaluate(filtered, sample_size=None, adapter_path=latest_adapter)
                report["adapter_path_used"] = latest_adapter
                report["golden_ids_path"] = str(golden_ids_path)
                write_json(golden_report_path, report)
                st.success(f"Complete — {report.get('eval_rows', 0)} rows evaluated.")
                _render_golden_results(report)

    if golden_report_path.exists():
        with st.expander("View cached Golden Eval report", expanded=False):
            cached = read_json(golden_report_path)
            _render_golden_results(cached)
            st.json(cached)


def _render_golden_results(report: dict) -> None:
    agg = report.get("aggregates", {})
    if not agg:
        st.warning("No aggregates in report.")
        return
    agg_rows = [{"variant": k, **v} for k, v in agg.items()]
    agg_df = pd.DataFrame(agg_rows)

    col_a, col_b = st.columns(2)
    with col_a:
        fig = _colored_bar(
            agg_df.rename(columns={"avg_format_adherence": "format_adherence"}),
            "format_adherence", "Avg Format Adherence", range_y=[0, 1],
        )
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig = _colored_bar(
            agg_df.rename(columns={"avg_latency_ms": "latency_ms"}),
            "latency_ms", "Avg Latency (ms)",
        )
        st.plotly_chart(fig, use_container_width=True)

    proxy_cols = {"avg_category_match", "avg_severity_match", "avg_lexical_overlap"}
    if proxy_cols.issubset(agg_df.columns):
        plot_df = agg_df.rename(columns={
            "avg_category_match": "category_match",
            "avg_severity_match": "severity_match",
            "avg_lexical_overlap": "lexical_overlap",
            "avg_format_adherence": "format_adherence",
        })
        st.plotly_chart(_radar_chart(plot_df), use_container_width=True)

    st.dataframe(
        agg_df.style.format({c: "{:.3f}" for c in agg_df.columns if c != "variant"})
        .set_properties(**{"background-color": "#1e293b", "color": "#cbd5e1"}),
        use_container_width=True,
    )


def main() -> None:
    settings = load_settings()
    st.set_page_config(
        page_title="Capstone 4 — Variant Compare",
        page_icon="⚡",
        layout="wide",
    )
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)

    latest_adapter = resolve_latest_adapter(settings.models_dir) or settings.adapter_path

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown(
        """
        <h1 style="font-size:1.8rem;margin-bottom:2px">
            ⚡ Capstone Project: Fine Tuning & RAG for Code Review Assistance
        </h1>
        <h2 style="font-size:1.8rem;margin-bottom:2px">
            🔬 Code Review Model Variant Evaluator
        </h2>
        <p style="margin-top:0;font-size:0.9rem">
            Side-by-side evaluation of Base · RAG · Fine-Tuned · Fine-Tuned+RAG
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ── Model & adapter info panel ────────────────────────────────────────────
    with st.expander("Model & Fine-Tuning Info", expanded=True):
        _render_model_info(settings, latest_adapter)

    # variant legend
    legend = "".join(_badge_html(v) for v in VARIANT_COLORS)
    st.markdown(f'<div style="margin:12px 0 8px">{legend}</div>', unsafe_allow_html=True)

    _render_previous_run_panel(settings)

    eval_path = settings.eval_file
    eval_rows: list[dict] = []
    if eval_path.exists():
        with eval_path.open("r", encoding="utf-8") as f:
            eval_rows = [json.loads(line) for line in f if line.strip()]

    default_prompt = (
        "Review this code change and return structured review comments "
        "with Severity, Category, Issue, WhyItMatters, SuggestedFix, and Confidence."
    )

    # ── Prompt panel ──────────────────────────────────────────────────────────
    st.markdown('<hr class="sec-divider">', unsafe_allow_html=True)
    st.markdown("### Prompt")

    col_prompt, col_run = st.columns([3, 1])
    with col_prompt:
        prompt_mode = st.radio(
            "Source", ["Custom prompt", "From eval dataset"],
            horizontal=True, label_visibility="collapsed",
        )
        selected_row = None
        if prompt_mode == "From eval dataset" and eval_rows:
            row_idx = st.slider("Eval row", 0, len(eval_rows) - 1, 0)
            selected_row = eval_rows[row_idx]
            prompt_text = _build_prompt(selected_row)
            st.text_area("Prompt preview", value=prompt_text, height=200)
            with st.expander("Expected output (ground truth)", expanded=False):
                st.code(_format_expected(selected_row), language="yaml")
        else:
            prompt_text = st.text_area("Prompt", value=default_prompt, height=200,
                                       label_visibility="collapsed")

    with col_run:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        run = st.button("▶ Run Comparison", use_container_width=True)

    # ── Comparison results ────────────────────────────────────────────────────
    if run:
        with st.spinner("Running 4 variants…"):
            router = InferenceRouter(adapter_path=latest_adapter)
            results = router.run_all_with_timings(prompt_text)

        metrics = [_metric_row(r, selected_row) for r in results]
        metric_df = pd.DataFrame(metrics)

        st.markdown('<hr class="sec-divider">', unsafe_allow_html=True)
        st.markdown("### Variant Outputs")
        _render_variant_grid(results)

        st.markdown('<hr class="sec-divider">', unsafe_allow_html=True)
        st.markdown("### Metrics")

        # inline metric chips per variant
        for r in metrics:
            v = r["variant"]
            color = _color(v)
            chips = "".join([
                f'<span class="mchip"><div class="mv">{r["format_adherence"]:.2f}</div><div class="mk">adherence</div></span>',
                f'<span class="mchip"><div class="mv">{r["latency_ms"]:.0f}</div><div class="mk">ms</div></span>',
                f'<span class="mchip"><div class="mv">{r["token_estimate"]}</div><div class="mk">tokens</div></span>',
            ])
            st.markdown(
                f'<div style="margin-bottom:10px">'
                f'{_badge_html(v)}'
                f'<div class="metric-row" style="margin-top:6px">{chips}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # charts
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                _colored_bar(metric_df, "format_adherence", "Format Adherence", [0, 1]),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                _colored_bar(metric_df, "latency_ms", "Latency (ms)"),
                use_container_width=True,
            )

        st.plotly_chart(
            _colored_bar(metric_df, "token_estimate", "Token Estimate"),
            use_container_width=True,
        )

        if "category_match" in metric_df.columns:
            st.plotly_chart(_radar_chart(metric_df), use_container_width=True)

        with st.expander("Raw metrics table", expanded=False):
            st.dataframe(metric_df, use_container_width=True)

    # ── Golden eval panel ────────────────────────────────────────────────────
    _render_golden_panel(settings, latest_adapter)


if __name__ == "__main__":
    main()
