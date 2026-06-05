# Evaluation Plan and Thresholds

## Comparison Variants
1. base
2. base_plus_rag
3. fine_tuned
4. fine_tuned_plus_rag

## Dataset
- Eval source: `data/raw/eval.jsonl`
- Current size: 40 rows

## Metrics
1. Format adherence (required sections present)
2. Category match (proxy)
3. Severity match (proxy)
4. Lexical overlap (proxy)
5. Latency (ms)
6. Token estimate

## Acceptance Targets
- Fine-tuned format adherence >= 0.95 target (production aspiration)
- Fine-tuned category/severity proxies should not regress against base
- Full report required for every model promotion

## Commands
Quick sample:
`python -m src.eval.run_eval --sample-size 5`

Full scorecard:
`python -m src.eval.run_eval --eval data/raw/eval.jsonl --report artifacts/reports/variant_eval_report_full.json`

## Reporting
- Reports written under `artifacts/reports/`
- Include `adapter_path_used` in each report
- Keep historical reports for trend analysis
