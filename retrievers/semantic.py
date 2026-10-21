"""Semantic retriever: sentence-transformers embeddings + cosine similarity (Hemakshi)."""
from pathlib import Path

import numpy as np

from common.base import Retriever
from common.config import EMBED_MODEL, MAX_SEQ_LEN

CHUNK = 20_000  # docs per checkpoint chunk while encoding


class SemanticRetriever(Retriever):
    name = "semantic"

    def __init__(self, model=None, batch_size: int = 128):
        self._model = model              # inject a stub in tests; None = load real model lazily
        self.batch_size = batch_size
        self.emb: np.ndarray | None = None   # (N, 384) float32, L2-normalised
        self.doc_ids: list[str] = []         # row i of emb belongs to doc_ids[i]
        self._pos: dict[str, int] = {}
        self._last_q: str | None = None      # one-entry query cache
        self._last_v: np.ndarray | None = None

    # ---------- encoding ----------
    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(EMBED_MODEL)
            self._model.max_seq_length = MAX_SEQ_LEN
        return self._model

    def _encode(self, texts: list[str], progress: bool = False) -> np.ndarray:
        v = self._get_model().encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=progress)
        return np.asarray(v, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Unit-length query vector (used by hybrid / Rocchio). Returns a copy."""
        if query != self._last_q:
            self._last_v, self._last_q = self._encode([query])[0], query
        return self._last_v.copy()

    # ---------- Retriever interface ----------
    def build(self, docs: dict[str, str], checkpoint_dir: Path | None = None) -> None:
        self.doc_ids = sorted(docs)
        texts = [docs[d] for d in self.doc_ids]
        parts = []
        for start in range(0, len(texts), CHUNK):
            f = checkpoint_dir / f"part_{start:08d}.npy" if checkpoint_dir else None
            if f is not None and f.exists():
                parts.append(np.load(f)); continue          # resume after a crash
            chunk = self._encode(texts[start:start + CHUNK], progress=True)
            if f is not None:
                checkpoint_dir.mkdir(parents=True, exist_ok=True); np.save(f, chunk)
            parts.append(chunk)
        self.emb = np.vstack(parts) if parts else np.zeros((0, 384), dtype=np.float32)
        self._pos = {d: i for i, d in enumerate(self.doc_ids)}

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        if self.emb is None or not (query or "").strip():
            return []
        return self.search_by_vector(self.embed_query(query), k)

    def search_by_vector(self, vec: np.ndarray, k: int = 100) -> list[tuple[str, float]]:
        """Cosine search with any query vector (e.g. a Rocchio-modified one)."""
        vec = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if self.emb is None or norm == 0:
            return []
        scores = self.emb @ (vec / norm)
        k = min(k, len(scores))
        if k <= 0:
            return []
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.doc_ids[i], float(scores[i])) for i in idx]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "emb.npy", self.emb)
        np.save(path / "doc_ids.npy", np.array(self.doc_ids))      # unicode array, no pickle

    def load(self, path: Path) -> None:
        self.emb = np.load(path / "emb.npy")
        self.doc_ids = np.load(path / "doc_ids.npy").tolist()
        self._pos = {d: i for i, d in enumerate(self.doc_ids)}

    def explain(self, query: str, doc_id: str) -> dict:
        if self.emb is None or doc_id not in self._pos:
            return super().explain(query, doc_id)
        cos = float(self.emb[self._pos[doc_id]] @ self.embed_query(query))
        return {"doc_id": doc_id, "score": cos, "matched_terms": [],
                "components": {"semantic": cos},
                "note": f"meaning-based match to the query (cosine similarity {cos:.2f})"}
