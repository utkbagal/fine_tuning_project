from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

from src.data.io_jsonl import read_json, write_json
from src.training.models import TrainingRunConfig, TrainingRunState, utc_now_iso


class TrainingRunManager:
    def __init__(self, checkpoints_dir: Path, models_dir: Path, reports_dir: Path) -> None:
        self.checkpoints_dir = checkpoints_dir
        self.models_dir = models_dir
        self.reports_dir = reports_dir

    def bootstrap(self, config: TrainingRunConfig) -> TrainingRunState:
        checkpoint_dir = self.checkpoints_dir / config.run_id
        adapter_output_dir = self.models_dir / config.run_id
        run_report_dir = self.reports_dir / config.run_id

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        adapter_output_dir.mkdir(parents=True, exist_ok=True)
        run_report_dir.mkdir(parents=True, exist_ok=True)

        state = TrainingRunState.init(
            run_id=config.run_id,
            checkpoint_dir=checkpoint_dir,
            adapter_output_dir=adapter_output_dir,
        )

        write_json(run_report_dir / "run_config.json", config.to_dict())
        write_json(run_report_dir / "run_state.json", state.to_dict())
        return state

    def run_report_dir(self, run_id: str) -> Path:
        return self.reports_dir / run_id

    def load_state(self, run_id: str) -> dict[str, Any]:
        return read_json(self.run_report_dir(run_id) / "run_state.json")

    def load_config(self, run_id: str) -> dict[str, Any]:
        return read_json(self.run_report_dir(run_id) / "run_config.json")

    def update_state(self, run_id: str, reports_dir: Path, status: str, metrics: dict[str, Any] | None = None, error: str = "") -> None:
        run_report_dir = reports_dir / run_id
        state_file = run_report_dir / "run_state.json"

        if not state_file.exists():
            raise FileNotFoundError(f"Missing state file: {state_file}")

        import json

        with state_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        payload["status"] = status
        payload["updated_at"] = utc_now_iso()
        if metrics is not None:
            payload["metrics"] = metrics
        if error:
            payload["error"] = error

        write_json(state_file, payload)

    def promote_latest_adapter(self, run_id: str) -> Path:
        run_state = self.load_state(run_id)
        source_dir = Path(run_state["adapter_output_dir"]).resolve()
        if not source_dir.exists():
            raise FileNotFoundError(f"Adapter output does not exist: {source_dir}")

        latest_dir = self.models_dir / "latest"
        latest_meta = self.models_dir / "latest.json"

        # Symlink can fail on Drive/Windows; fallback to directory copy.
        # Cleanup is done inside the try so that if rmtree also fails (e.g. Drive
        # Errno 95), copytree can still succeed via dirs_exist_ok=True.
        try:
            if latest_dir.exists() or latest_dir.is_symlink():
                if latest_dir.is_symlink() or latest_dir.is_file():
                    latest_dir.unlink()
                else:
                    shutil.rmtree(latest_dir)
            latest_dir.symlink_to(source_dir, target_is_directory=True)
            mode = "symlink"
        except OSError:
            # dirs_exist_ok=True lets copytree overwrite an existing latest dir
            # without requiring a prior successful rmtree (critical for Drive FUSE).
            shutil.copytree(source_dir, latest_dir, dirs_exist_ok=True)
            mode = "copy"

        write_json(
            latest_meta,
            {
                "run_id": run_id,
                "source_dir": str(source_dir),
                "latest_dir": str(latest_dir),
                "mode": mode,
                "updated_at": utc_now_iso(),
            },
        )
        return latest_dir
