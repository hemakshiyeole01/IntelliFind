from abc import ABC, abstractmethod
from pathlib import Path


class Retriever(ABC):
    name: str = "base"          # lowercase, unique: tfidf | bm25 | semantic | hybrid

    @abstractmethod
    def build(self, docs: dict[str, str]) -> None:
        """Build index from {doc_id: doc_text}. Called once, offline."""

    @abstractmethod
    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        """Return up to k (doc_id, raw_score), sorted by score DESC (higher = better)."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Write everything needed by search() into the folder `path`."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Restore from `path`. After load(), search() must work WITHOUT build()."""

    def explain(self, query: str, doc_id: str) -> dict:
        """Used by the UI. Override to add real details."""
        return {"doc_id": doc_id, "score": None, "matched_terms": [], "components": {}, "note": ""}
