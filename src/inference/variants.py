from __future__ import annotations

from dataclasses import dataclass

from src.inference.backends import TextGenerationBackend
from src.inference.types import InferenceInput, InferenceOutput
from src.rag.retriever import KeywordRetriever


@dataclass
class BaseVariant:
    backend: TextGenerationBackend
    name: str = "base"

    def run(self, request: InferenceInput, max_new_tokens: int = 384) -> InferenceOutput:
        out = self.backend.generate(request.prompt, max_new_tokens=max_new_tokens)
        return InferenceOutput(variant=self.name, text=out)


@dataclass
class BaseRagVariant:
    backend: TextGenerationBackend
    retriever: KeywordRetriever
    name: str = "base_plus_rag"

    def run(self, request: InferenceInput, max_new_tokens: int = 384) -> InferenceOutput:
        contexts = self.retriever.retrieve(request.prompt, top_k=3)
        context_block = "\n\n".join([f"Source: {c.source}\n{c.content[:700]}" for c in contexts])
        rag_prompt = request.prompt
        if context_block:
            rag_prompt = f"Use context to answer.\n\n{context_block}\n\nUser request:\n{request.prompt}"
        out = self.backend.generate(rag_prompt, max_new_tokens=max_new_tokens)
        return InferenceOutput(
            variant=self.name,
            text=out,
            metadata={"retrieved_sources": [c.source for c in contexts]},
        )


@dataclass
class FineTunedVariant:
    backend: TextGenerationBackend
    adapter_path: str
    name: str = "fine_tuned"

    def run(self, request: InferenceInput, max_new_tokens: int = 384) -> InferenceOutput:
        out = self.backend.generate(request.prompt, max_new_tokens=max_new_tokens)
        return InferenceOutput(
            variant=self.name,
            text=out,
            metadata={"adapter_path": self.adapter_path},
        )


@dataclass
class FineTunedRagVariant:
    backend: TextGenerationBackend
    retriever: KeywordRetriever
    adapter_path: str
    name: str = "fine_tuned_plus_rag"

    def run(self, request: InferenceInput, max_new_tokens: int = 384) -> InferenceOutput:
        contexts = self.retriever.retrieve(request.prompt, top_k=3)
        context_block = "\n\n".join([f"Source: {c.source}\n{c.content[:700]}" for c in contexts])
        rag_prompt = request.prompt
        if context_block:
            rag_prompt = (
                f"Use context to answer.\n\n{context_block}\n\nUser request:\n{request.prompt}"
            )
        out = self.backend.generate(rag_prompt, max_new_tokens=max_new_tokens)
        return InferenceOutput(
            variant=self.name,
            text=out,
            metadata={
                "adapter_path": self.adapter_path,
                "retrieved_sources": [c.source for c in contexts],
            },
        )
