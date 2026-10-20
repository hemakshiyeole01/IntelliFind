from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from common.base import Retriever
from common.config import BM25_B, BM25_K1

from .index import InvertedIndex, top_k


class Bm25Retriever(Retriever):
    """idf = log(1 + (N-df+0.5)/(df+0.5));
    score = sum idf * tf*(k1+1) / (tf + k1*(1 - b + b*len/avglen)). Raw scores."""

    name = "bm25"

    def __init__(self, index: InvertedIndex | None = None,
                 k1: float = BM25_K1, b: float = BM25_B) -> None:
        self.index = index  # pass an already-built index to share it with tfidf
        self.k1, self.b = k1, b
        self._K: np.ndarray | None = None

    # ---- build / persistence ----
    def build(self, docs: dict[str, str]) -> None:
        if self.index is None:
            self.index = InvertedIndex()
            self.index.build(docs)
        self._prepare()

    def _prepare(self) -> None:
        ix = self.index
        self._K = self.k1 * (1.0 - self.b + self.b * ix.doc_len_arr / ix.avglen)

    def save(self, path: Path) -> None:
        self.index.save(Path(path))

    def load(self, path: Path) -> None:
        self.index = InvertedIndex()
        self.index.load(Path(path))
        self._prepare()

    # ---- search ----
    def _score(self, query: str):
        ix = self.index
        terms = ix.query_terms(query)
        if not terms:
            return None
        scores = np.zeros(ix.N)
        cands = []
        for t in terms:
            df = ix.df[t]
            idf = math.log(1.0 + (ix.N - df + 0.5) / (df + 0.5))
            d, tf = ix.post_docs[t], ix.post_tf[t]
            scores[d] += idf * tf * (self.k1 + 1.0) / (tf + self._K[d])
            cands.append(d)
        return scores, np.unique(np.concatenate(cands))

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        res = self._score(query)
        if res is None:
            return []
        scores, cand = res
        return top_k(self.index, scores, cand, k)

    def explain(self, query: str, doc_id: str) -> dict:
        ix = self.index
        di = ix.id2idx.get(doc_id)
        if di is None:
            return {"doc_id": doc_id, "score": None, "matched_terms": [], "components": {},
                    "note": "Unknown document."}
        terms = ix.query_terms(query)
        matched = [t for t in terms if ix.tf_of(t, di) > 0]
        res = self._score(query)
        score = float(res[0][di]) if res is not None else 0.0
        note = (f"BM25 score {score:.3f}; matched {len(matched)} of {len(terms)} query terms"
                + (f" ({', '.join(matched)})." if matched else "."))
        return {"doc_id": doc_id, "score": score, "matched_terms": matched,
                "components": {"bm25": score}, "note": note}