from __future__ import annotations

from pathlib import Path

from src.data.io_jsonl import read_json


def resolve_latest_adapter(models_dir: Path) -> str | None:
    """Resolve latest adapter directory from artifacts metadata.

    Priority:
    1. artifacts/models/latest.json -> latest_dir
    2. artifacts/models/latest directory
    """
    latest_meta = models_dir / "latest.json"
    if latest_meta.exists():
        payload = read_json(latest_meta)
        latest_dir = Path(str(payload.get("latest_dir", "")))
        if latest_dir.exists() and latest_dir.is_dir():
            return str(latest_dir)

    latest_dir = models_dir / "latest"
    if latest_dir.exists() and latest_dir.is_dir():
        return str(latest_dir)

    return None
