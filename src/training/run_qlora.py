from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.config.settings import load_settings
from src.data.io_jsonl import read_jsonl, write_json
from src.training.formatting import build_training_text
from src.training.models import TrainingRunConfig
from src.training.run_manager import TrainingRunManager


def _module_exists(module_names: list[str], target: str) -> bool:
    return any(name == target or name.endswith(f".{target}") for name in module_names)


def _resolve_target_modules(model: Any, requested: tuple[str, ...]) -> tuple[str, ...]:
    module_names = [name for name, _ in model.named_modules()]
    requested_present = tuple(m for m in requested if _module_exists(module_names, m))
    if requested_present:
        return requested_present

    # Fallback sets for common decoder architectures when requested modules are absent.
    fallback_sets = [
        ("qkv_proj", "o_proj"),  # Phi-style attention blocks
        ("q_proj", "k_proj", "v_proj", "o_proj"),  # Llama/Mistral-style
        ("query_key_value",),  # Some GPT-NeoX-style blocks
        ("Wqkv",),  # Falcon-style blocks
        ("c_attn",),  # GPT-2 style blocks
    ]
    for candidate in fallback_sets:
        present = tuple(m for m in candidate if _module_exists(module_names, m))
        if present:
            return present

    raise ValueError(
        "Unable to resolve LoRA target modules for this model. "
        f"Requested: {requested}. "
        "Please set target modules compatible with the model architecture."
    )


def _prepare_text_rows(rows: list[dict[str, Any]], max_samples: int | None = None) -> list[dict[str, str]]:
    limited = rows if max_samples is None else rows[:max_samples]
    return [{"text": build_training_text(row)} for row in limited]


def _write_adapter_manifest(adapter_dir: Path, payload: dict[str, Any]) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    write_json(adapter_dir / "adapter_manifest.json", payload)


def _run_training(
    config: TrainingRunConfig,
    run_manager: TrainingRunManager,
    execute_training: bool,
    max_train_samples: int | None,
    hf_token: str,
    local_files_only: bool,
) -> dict[str, Any]:
    train_rows = read_jsonl(Path(config.train_file))
    eval_rows = read_jsonl(Path(config.eval_file))

    prepared_train = _prepare_text_rows(train_rows, max_samples=max_train_samples)
    prepared_eval = _prepare_text_rows(eval_rows, max_samples=min(len(eval_rows), 40))

    run_state = run_manager.load_state(config.run_id)
    adapter_dir = Path(run_state["adapter_output_dir"])

    metrics: dict[str, Any] = {
        "mode": "execute" if execute_training else "dry_run",
        "train_rows": len(prepared_train),
        "eval_rows": len(prepared_eval),
        "base_model_id": config.base_model_id,
        "hyperparameters": config.to_dict()["hyperparameters"],
    }

    if not execute_training:
        _write_adapter_manifest(
            adapter_dir,
            {
                "run_id": config.run_id,
                "mode": "dry_run",
                "message": "Placeholder adapter artifact for pipeline validation.",
                "base_model_id": config.base_model_id,
                "hyperparameters": config.to_dict()["hyperparameters"],
            },
        )
        return metrics

    # Stub triton.ops before importing bitsandbytes/transformers to prevent
    # ModuleNotFoundError on triton 3.x which removed the ops submodule.
    import sys as _sys
    import types as _types
    if "triton.ops" not in _sys.modules:
        try:
            import triton as _triton
            _stub = _types.ModuleType("triton.ops")
            _sys.modules["triton.ops"] = _stub
            if not hasattr(_triton, "ops"):
                _triton.ops = _stub  # type: ignore[attr-defined]
        except ImportError:
            pass

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_id,
        token=hf_token or None,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    if torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_id,
        token=hf_token or None,
        local_files_only=local_files_only,
        dtype=torch_dtype,
        device_map="auto",
        quantization_config=quantization_config,
    )

    resolved_target_modules = _resolve_target_modules(model, config.hyperparameters.target_modules)
    metrics["resolved_target_modules"] = list(resolved_target_modules)
    print("LoRA target modules:", ", ".join(resolved_target_modules))

    lora_cfg = LoraConfig(
        r=config.hyperparameters.rank,
        lora_alpha=config.hyperparameters.alpha,
        target_modules=list(resolved_target_modules),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    train_ds = Dataset.from_list(prepared_train)
    eval_ds = Dataset.from_list(prepared_eval)

    def tokenize_fn(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=1024)

    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    eval_tok = eval_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    output_dir = Path(run_state["checkpoint_dir"]) / "trainer_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.hyperparameters.batch_size,
        gradient_accumulation_steps=config.hyperparameters.gradient_accumulation_steps,
        learning_rate=config.hyperparameters.learning_rate,
        num_train_epochs=config.hyperparameters.epochs,
        logging_steps=10,
        save_steps=50,
        eval_strategy="no",
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=collator,
    )

    trainer_result = trainer.train()

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    metrics["train_loss"] = float(getattr(trainer_result, "training_loss", 0.0))
    metrics["global_step"] = int(getattr(trainer_result, "global_step", 0))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QLoRA training pipeline (dry-run by default).")
    parser.add_argument("--notes", default="", help="Optional run notes")
    parser.add_argument(
        "--execute-training",
        action="store_true",
        help="Execute actual training. Without this flag, script performs dry-run and writes placeholder adapter artifacts.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional cap for training rows")
    args = parser.parse_args()

    settings = load_settings()
    manager = TrainingRunManager(
        checkpoints_dir=settings.checkpoints_dir,
        models_dir=settings.models_dir,
        reports_dir=settings.reports_dir,
    )

    config = TrainingRunConfig.new(
        base_model_id=settings.model_source,
        data_version=settings.data_version,
        train_file=settings.train_file,
        eval_file=settings.eval_file,
        notes=args.notes,
    )
    manager.bootstrap(config)

    try:
        manager.update_state(config.run_id, settings.reports_dir, status="running")
        metrics = _run_training(
            config=config,
            run_manager=manager,
            execute_training=args.execute_training,
            max_train_samples=args.max_train_samples,
            hf_token=settings.hf_token,
            local_files_only=settings.hf_local_files_only,
        )
        manager.update_state(config.run_id, settings.reports_dir, status="completed", metrics=metrics)
        latest_dir = manager.promote_latest_adapter(config.run_id)
    except Exception as exc:
        manager.update_state(config.run_id, settings.reports_dir, status="failed", error=str(exc))
        raise

    print(f"Run completed: {config.run_id}")
    print(f"Mode: {'execute' if args.execute_training else 'dry_run'}")
    print(f"Latest adapter path: {latest_dir}")


if __name__ == "__main__":
    main()
