from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency in some runtimes
    load_dotenv = None


@dataclass(frozen=True)
class ProjectSettings:
    project_root: Path
    data_raw_dir: Path
    data_processed_dir: Path
    data_rag_dir: Path
    artifacts_dir: Path
    checkpoints_dir: Path
    models_dir: Path
    reports_dir: Path
    base_model_id: str
    model_base_path: str
    model_source: str
    use_hf_backend: bool
    hf_token: str
    hf_local_files_only: bool
    adapter_path: str
    max_new_tokens: int
    data_version: str
    train_file: Path
    eval_file: Path


def load_settings() -> ProjectSettings:
    root = Path(__file__).resolve().parents[2]
    if load_dotenv is not None:
        load_dotenv(root / ".env", override=False)

    use_hf_backend = os.getenv("USE_HF_BACKEND", "0").strip().lower() in {"1", "true", "yes"}
    hf_local_files_only = os.getenv("HF_LOCAL_FILES_ONLY", "0").strip().lower() in {"1", "true", "yes"}
    base_model_id = os.getenv("MODEL_BASE_ID", "microsoft/Phi-3.5-mini-instruct")
    model_base_path = os.getenv("MODEL_BASE_PATH", "").strip()

    # Auto-discover a local model folder when MODEL_BASE_PATH is not set.
    if not model_base_path:
        model_folder_name = base_model_id.split("/")[-1]
        candidates = [
            root / model_folder_name,
            root.parent / model_folder_name,
            root / "models" / model_folder_name,
            root.parent / "models" / model_folder_name,
        ]
        for candidate in candidates:
            if candidate.is_dir() and (candidate / "config.json").exists():
                model_base_path = str(candidate)
                break

    model_source = model_base_path or base_model_id
    return ProjectSettings(
        project_root=root,
        data_raw_dir=root / "data" / "raw",
        data_processed_dir=root / "data" / "processed",
        data_rag_dir=root / "data" / "rag_corpus",
        artifacts_dir=root / "artifacts",
        checkpoints_dir=root / "artifacts" / "checkpoints",
        models_dir=root / "artifacts" / "models",
        reports_dir=root / "artifacts" / "reports",
        base_model_id=base_model_id,
        model_base_path=model_base_path,
        model_source=model_source,
        use_hf_backend=use_hf_backend,
        hf_token=os.getenv("HF_TOKEN", ""),
        hf_local_files_only=hf_local_files_only,
        adapter_path=os.getenv("ADAPTER_PATH", "artifacts/models/latest"),
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "384")),
        data_version=os.getenv("DATA_VERSION", "v1.0"),
        train_file=root / "data" / "raw" / "train.jsonl",
        eval_file=root / "data" / "raw" / "eval.jsonl",
    )
