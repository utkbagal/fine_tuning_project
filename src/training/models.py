from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class LoRAHyperParameters:
    rank: int = 16
    alpha: int = 32
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    learning_rate: float = 2e-4
    epochs: int = 2
    batch_size: int = 4
    gradient_accumulation_steps: int = 16


@dataclass
class TrainingRunConfig:
    run_id: str
    run_name: str
    base_model_id: str
    data_version: str
    train_file: str
    eval_file: str
    created_at: str = field(default_factory=utc_now_iso)
    trainer: str = "qlora_peft_trl"
    notes: str = ""
    hyperparameters: LoRAHyperParameters = field(default_factory=LoRAHyperParameters)

    @staticmethod
    def new(base_model_id: str, data_version: str, train_file: Path, eval_file: Path, notes: str = "") -> "TrainingRunConfig":
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        return TrainingRunConfig(
            run_id=run_id,
            run_name=f"exp_{run_id}",
            base_model_id=base_model_id,
            data_version=data_version,
            train_file=str(train_file),
            eval_file=str(eval_file),
            notes=notes,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["hyperparameters"]["target_modules"] = list(self.hyperparameters.target_modules)
        return data


@dataclass
class TrainingRunState:
    run_id: str
    status: str
    created_at: str
    updated_at: str
    checkpoint_dir: str
    adapter_output_dir: str
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @staticmethod
    def init(run_id: str, checkpoint_dir: Path, adapter_output_dir: Path) -> "TrainingRunState":
        now = utc_now_iso()
        return TrainingRunState(
            run_id=run_id,
            status="created",
            created_at=now,
            updated_at=now,
            checkpoint_dir=str(checkpoint_dir),
            adapter_output_dir=str(adapter_output_dir),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
