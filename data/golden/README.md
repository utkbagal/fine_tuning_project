# Golden Set Notes

This folder stores immutable golden evaluation assets used for release gating.

Recommended contents:

- `golden_eval_ids.json`: fixed eval row ids used for trend comparisons.
- `golden_expected_notes.md`: optional human-reviewed expectations for difficult rows.

Current project uses `data/raw/eval.jsonl` as the held-out evaluation source.
If you create a strict golden subset for governance gates, place it here and keep it versioned.

## How golden_eval_ids.json is used

The eval runner supports filtering eval rows by a stable golden-id list:

- Command:
	`python -m src.eval.run_eval --eval data/raw/eval.jsonl --golden-ids data/golden/golden_eval_ids.json --report artifacts/reports/variant_eval_report_golden.json`

- Behavior:
	- Reads ids from `ids` field in `golden_eval_ids.json`
	- Filters `eval.jsonl` rows to only matching ids
	- Runs all four variants on that fixed subset
	- Writes `golden_ids_path` in the output report for traceability
