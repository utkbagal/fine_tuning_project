from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RetrievedContext:
    source: str
    content: str


class KeywordRetriever:
    """Very small retrieval baseline for early development.

    It scans .md/.txt files in the RAG corpus and returns top-k contexts
    by keyword overlap. This is intentionally simple for sprint speed.
    """

    def __init__(self, corpus_dir: Path) -> None:
        self.corpus_dir = corpus_dir
        self.docs: list[RetrievedContext] = []
        self._load()

    def _load(self) -> None:
        if not self.corpus_dir.exists():
            return
        for p in self.corpus_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".md", ".txt"}:
                self.docs.append(RetrievedContext(source=str(p), content=p.read_text(encoding="utf-8", errors="ignore")))

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedContext]:
        if not self.docs:
            return []
        q_terms = {t for t in query.lower().split() if len(t) > 2}
        scored: list[tuple[int, RetrievedContext]] = []
        for doc in self.docs:
            text = doc.content.lower()
            score = sum(1 for t in q_terms if t in text)
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_k]]
