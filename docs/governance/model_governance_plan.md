# Model Governance Plan

## Scope
This plan governs dataset lifecycle, model training, adapter promotion, inference usage, and retraining triggers for the code-review assistant.

## Versioning
- Dataset versions: semantic tag in metadata (for example `v1.0`)
- Training runs: unique run_id (timestamp + suffix)
- Adapter artifacts: `artifacts/models/<run_id>/`
- Active adapter alias: `artifacts/models/latest`

## Change Control
- Any dataset schema or rubric change requires a new data version
- Any hyperparameter change requires a new run_id and report
- Adapter promotion to latest requires completed run state

## Monitoring KPIs
1. Format adherence
2. Category match rate
3. Severity match rate
4. Lexical overlap proxy
5. Latency and token estimate

## Deployment Policy
- Base and fine-tuned backends are both deployable
- Fine-tuned deployment requires successful backend verification report
- Rollback path: point `ADAPTER_PATH` to previous run artifact

## Retraining Triggers
- Format adherence drops below threshold
- Category/severity proxy metrics degrade over two consecutive eval cycles
- New review policy categories introduced
- Significant drift in code patterns/domains

## Auditability
- Preserve run_config, run_state, and evaluation reports per run
- Keep backend verification reports with adapter path references
- Store ADR and failure mode register alongside model lifecycle documents
