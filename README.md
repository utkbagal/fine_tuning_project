# Capstone 4 - Code Review Fine-Tuning Project

This project implements Capstone 4 end-to-end for domain-specialized code review generation.

## Scope

- Domain: Code review comments
- Base model target: `microsoft/Phi-3.5-mini-instruct`
- Fine-tuning approach: QLoRA
- Required comparisons:
  - Base
  - Base + RAG
  - Fine-tuned
  - Fine-tuned + RAG

## Quick Start

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy generated dataset files into `data/raw/`:
- `train.jsonl`
- `eval.jsonl`

4. Run validation:

```bash
python -m src.data.validate_dataset --train data/raw/train.jsonl --eval data/raw/eval.jsonl --report artifacts/reports/dataset_validation_report.json
```

5. Create a training run scaffold:

```bash
python -m src.training.start_run --notes "initial run"
```

6. Run QLoRA pipeline (dry-run default, creates adapter artifacts + updates latest pointer):

```bash
python -m src.training.run_qlora --notes "dry-run validation"
```

For actual training execution on GPU-enabled environment:

```bash
python -m src.training.run_qlora --execute-training --notes "colab run"
```

7. Run variant evaluation (quick sample):

```bash
python -m src.eval.run_eval --eval data/raw/eval.jsonl --report artifacts/reports/variant_eval_report.json --sample-size 5
```

8. Run full eval on all rows:

```bash
python -m src.eval.run_eval --eval data/raw/eval.jsonl --report artifacts/reports/variant_eval_report_full.json
```

9. Run golden-only eval (fixed subset for trend comparisons):

```bash
python -m src.eval.run_eval --eval data/raw/eval.jsonl --golden-ids data/golden/golden_eval_ids.json --report artifacts/reports/variant_eval_report_golden.json
```

10. Verify backend loading and adapter path wiring:

```bash
python -m src.inference.verify_backend --report artifacts/reports/backend_verification.json
```

11. Launch interactive comparison UI:

```bash
streamlit run src/app/compare_app.py
```

By default, eval auto-resolves the newest adapter from `artifacts/models/latest.json` (or `artifacts/models/latest`).
You can override it with:

```bash
python -m src.eval.run_eval --adapter-path artifacts/models/<run_id>
```

## Colab Notebook

Use [project/notebooks/colab/capstone4_training_runbook.ipynb](notebooks/colab/capstone4_training_runbook.ipynb) for GPU execution in Google Colab.
It includes:
- environment setup
- dataset validation
- real QLoRA training execution
- backend verification
- full variant eval command sequence
- golden-only eval command
- notebook-native 4-variant report charts
- notebook-native single-prompt 4-variant comparison

## Hugging Face Backend Toggle

Inference is wired for real model execution. If configured source loading fails,
the system attempts local base-model fallbacks automatically.

- To enable real HF inference, set:
  - `USE_HF_BACKEND=1`
  - `MODEL_BASE_ID=microsoft/Phi-3.5-mini-instruct`
  - `MODEL_BASE_PATH=<optional_local_model_dir>`
  - `HF_LOCAL_FILES_ONLY=1` (use cache/local only; no download)
  - `HF_TOKEN=<optional_hf_token_if_remote_download_is_needed>`
  - `ADAPTER_PATH=<path_to_lora_adapter_dir>`

With HF backend enabled, the inference router loads model weights via `transformers`.
If `ADAPTER_PATH` exists and includes valid PEFT adapter files, fine-tuned variants load adapter weights on top of the base model.
If adapter artifacts are not ready, base variants still run and fine-tuned variants return explicit status messages.

## RAG Corpus and Data Folders

- `data/rag_corpus/` contains seed retrieval docs used by `KeywordRetriever`.
- `data/golden/` documents golden-set conventions for release gating assets.
- `data/golden/golden_eval_ids.json` defines fixed eval IDs for golden-only scoring.
- `data/processed/` documents intended intermediate preprocessing outputs.

## Repository Layout

- `src/config/` shared settings
- `src/data/` data IO and validation
- `src/training/` training orchestration and adapter promotion
- `src/inference/` base/fine-tuned inference wrappers and fallback handling
- `src/eval/` scoring harness and golden-id filtering
- `src/app/` comparison UI with prompt compare and golden eval panel
- `artifacts/reports/` generated reports

## Status

Implementation complete for capstone coding scope:

- data validation and IO pipeline
- QLoRA training orchestration and adapter promotion
- resilient base/fine-tuned inference backends with fallback handling
- 4-variant router and evaluation harness
- Streamlit comparison UI and notebook equivalent
- Colab GPU runbook with notebook-native 4-variant analysis cells
- governance docs: ADR, failure mode register, trust boundary canvas, governance plan

Remaining non-code deliverables are execution/reporting activities (final training run evidence and presentation deck).
