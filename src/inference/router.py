from __future__ import annotations

from dataclasses import asdict
from time import perf_counter

from src.config.settings import load_settings
from src.inference.backends import create_backend, create_fine_tuned_backend
from src.inference.types import InferenceInput
from src.inference.variants import BaseRagVariant, BaseVariant, FineTunedRagVariant, FineTunedVariant
from src.rag.retriever import KeywordRetriever


class InferenceRouter:
    def __init__(self, adapter_path: str | None = None) -> None:
        settings = load_settings()
        retriever = KeywordRetriever(settings.data_rag_dir)
        base_backend = create_backend(
            model_id=settings.model_source,
            use_hf_backend=settings.use_hf_backend,
            hf_token=settings.hf_token,
            local_files_only=settings.hf_local_files_only,
        )
        self.max_new_tokens = settings.max_new_tokens
        resolved_adapter_path = adapter_path or settings.adapter_path
        fine_tuned_backend = create_fine_tuned_backend(
            model_id=settings.model_source,
            use_hf_backend=settings.use_hf_backend,
            adapter_path=resolved_adapter_path,
            hf_token=settings.hf_token,
            local_files_only=settings.hf_local_files_only,
        )

        self.variants = [
            BaseVariant(backend=base_backend),
            BaseRagVariant(backend=base_backend, retriever=retriever),
            FineTunedVariant(backend=fine_tuned_backend, adapter_path=resolved_adapter_path),
            FineTunedRagVariant(backend=fine_tuned_backend, retriever=retriever, adapter_path=resolved_adapter_path),
        ]

    def run_all(self, prompt: str) -> list[dict]:
        request = InferenceInput(prompt=prompt)
        return [asdict(v.run(request, max_new_tokens=self.max_new_tokens)) for v in self.variants]

    def run_all_with_timings(self, prompt: str) -> list[dict]:
        request = InferenceInput(prompt=prompt)
        results: list[dict] = []
        for variant in self.variants:
            start = perf_counter()
            output = variant.run(request, max_new_tokens=self.max_new_tokens)
            elapsed_ms = round((perf_counter() - start) * 1000, 2)
            payload = asdict(output)
            payload["latency_ms"] = elapsed_ms
            results.append(payload)
        return results


def demo(prompt: str) -> None:
    router = InferenceRouter()
    results = router.run_all_with_timings(prompt)
    for result in results:
        print(f"=== {result['variant']} ===")
        print(result["text"])
        print(f"latency_ms: {result['latency_ms']}")
        if result.get("metadata"):
            print(f"metadata: {result['metadata']}")
        print()


if __name__ == "__main__":
    demo("Review this API auth code and provide a structured review comment.")
