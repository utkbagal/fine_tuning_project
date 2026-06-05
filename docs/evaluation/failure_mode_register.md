# Failure Mode Register (FMEA)

| ID | Failure Mode | Cause | Effect | Detection | Mitigation | Severity | Occurrence | Detection Score |
|---|---|---|---|---|---|---:|---:|---:|
| FM-01 | Overfitting to fixed response templates | Low data diversity | Generic/rigid responses | Eval lexical overlap drift, human review | Increase scenario diversity, add hard negatives | 8 | 5 | 4 |
| FM-02 | Category misclassification | Ambiguous snippets | Incorrect review routing | Category match metric | Add targeted category examples, rebalance data | 7 | 5 | 5 |
| FM-03 | Severity inflation | Training bias toward high severity | Alarm fatigue for reviewers | Severity distribution monitoring | Rebalance severity labels, recalibrate prompts | 6 | 6 | 4 |
| FM-04 | Severity underestimation | Missing critical edge cases | High-risk issues downplayed | Edge-case eval pack | Add security/critical edge-case augmentations | 9 | 4 | 5 |
| FM-05 | Adapter load failure in runtime | Path mismatch or missing files | Fine-tuned variant unavailable | Backend verification report | Auto-resolve latest + fallback + deployment checks | 7 | 3 | 2 |
| FM-06 | Retrieval contamination | Low-quality RAG corpus | Wrong contextual guidance | Source inspection + eval deltas | Curate corpus, add source constraints | 6 | 4 | 5 |
| FM-07 | Latency spikes | Large context + model load contention | Poor user experience | Latency KPI monitoring | Token budget, caching, batch strategy | 5 | 5 | 3 |
| FM-08 | Catastrophic forgetting | Aggressive finetuning params | Base reasoning capability loss | Compare base vs tuned on control set | Conservative epochs/LR, hold-out controls | 8 | 3 | 6 |
| FM-09 | Bias in recommendations | Skewed synthetic data | Unfair or inconsistent guidance | Bias probe set | Expand diverse examples, review rubric | 8 | 4 | 6 |

## Notes
- Recompute risk prioritization each training cycle.
- Tie remediation actions to run_id and re-evaluate after each mitigation.
