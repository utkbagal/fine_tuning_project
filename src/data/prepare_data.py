from __future__ import annotations

import argparse
from pathlib import Path

from src.config.settings import load_settings
from src.data.io_jsonl import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy external train/eval JSONL into project data/raw.")
    parser.add_argument("--source-train", required=True, type=Path)
    parser.add_argument("--source-eval", required=True, type=Path)
    args = parser.parse_args()

    settings = load_settings()
    settings.data_raw_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(args.source_train)
    eval_rows = read_jsonl(args.source_eval)

    train_target = settings.data_raw_dir / "train.jsonl"
    eval_target = settings.data_raw_dir / "eval.jsonl"

    write_jsonl(train_target, train_rows)
    write_jsonl(eval_target, eval_rows)

    print(f"Copied train: {len(train_rows)} -> {train_target}")
    print(f"Copied eval: {len(eval_rows)} -> {eval_target}")


if __name__ == "__main__":
    main()
