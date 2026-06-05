from __future__ import annotations

import argparse

from src.config.settings import load_settings
from src.training.models import TrainingRunConfig
from src.training.run_manager import TrainingRunManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a training run scaffold and metadata artifacts.")
    parser.add_argument("--notes", default="", help="Optional run notes.")
    args = parser.parse_args()

    settings = load_settings()

    config = TrainingRunConfig.new(
        base_model_id=settings.model_source,
        data_version=settings.data_version,
        train_file=settings.train_file,
        eval_file=settings.eval_file,
        notes=args.notes,
    )

    manager = TrainingRunManager(
        checkpoints_dir=settings.checkpoints_dir,
        models_dir=settings.models_dir,
        reports_dir=settings.reports_dir,
    )
    state = manager.bootstrap(config)

    print(f"Run created: {config.run_id}")
    print(f"Run name: {config.run_name}")
    print(f"Status: {state.status}")
    print(f"Checkpoint dir: {state.checkpoint_dir}")
    print(f"Adapter output dir: {state.adapter_output_dir}")


if __name__ == "__main__":
    main()
