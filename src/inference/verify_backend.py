from __future__ import annotations

import argparse
from pathlib import Path

from src.config.settings import load_settings
from src.data.io_jsonl import write_json
from src.inference.adapter_registry import resolve_latest_adapter
from src.inference.backends import create_backend, create_fine_tuned_backend


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify base and fine-tuned inference backends.")
    parser.add_argument("--run-generate", action="store_true", help="Attempt a tiny generation call for each backend")
    parser.add_argument("--report", type=Path, default=None, help="Output report path")
    args = parser.parse_args()

    settings = load_settings()
    adapter_path = resolve_latest_adapter(settings.models_dir) or settings.adapter_path

    base_backend = create_backend(
        model_id=settings.model_source,
        use_hf_backend=settings.use_hf_backend,
        hf_token=settings.hf_token,
        local_files_only=settings.hf_local_files_only,
    )
    ft_backend = create_fine_tuned_backend(
        model_id=settings.model_source,
        use_hf_backend=settings.use_hf_backend,
        adapter_path=adapter_path,
        hf_token=settings.hf_token,
        local_files_only=settings.hf_local_files_only,
    )

    report = {
        "base_model_id": settings.base_model_id,
        "model_source": settings.model_source,
        "use_hf_backend": settings.use_hf_backend,
        "hf_local_files_only": settings.hf_local_files_only,
        "adapter_path": adapter_path,
        "base_backend_class": base_backend.__class__.__name__,
        "fine_tuned_backend_class": ft_backend.__class__.__name__,
        "generation": {},
    }

    if args.run_generate:
        prompt = "Provide one concise code review comment with Severity and Issue fields."
        for name, backend in (("base", base_backend), ("fine_tuned", ft_backend)):
            try:
                out = backend.generate(prompt, max_new_tokens=64)
                report["generation"][name] = {
                    "ok": True,
                    "preview": out[:300],
                }
            except Exception as exc:
                report["generation"][name] = {
                    "ok": False,
                    "error": str(exc),
                }

    report_path = args.report or (settings.reports_dir / "backend_verification.json")
    write_json(report_path, report)

    print(f"Base backend: {report['base_backend_class']}")
    print(f"Fine-tuned backend: {report['fine_tuned_backend_class']}")
    print(f"Adapter path: {adapter_path}")
    print(f"Report written: {report_path}")


if __name__ == "__main__":
    main()
