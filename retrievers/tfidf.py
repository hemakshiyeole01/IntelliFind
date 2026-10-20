from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np

from common.base import Retriever
from common.text import tokenize

from .index import InvertedIndex, top_k


class TfidfRetriever(Retriever):
    """weight = (1 + log tf) * log(N / df); cosine for documents and query."""

    name = "tfidf"
    NORMS_FILE = "tfidf_norms.npy"

    def __init__(self, index: InvertedIndex | None = None) -> None:
        self.index = index  # pass an already-built index to share it with bm25
        self._norms: np.ndarray | None = None

    # ---- build / persistence ----
    def build(self, docs: dict[str, str]) -> None:
        if self.index is None:
            self.index = InvertedIndex()
            self.index.build(docs)
        self._compute_norms()

    def _compute_norms(self) -> None:
        ix = self.index
        sq = np.zeros(ix.N)
        for t, d in ix.post_docs.items():
            idf = math.log(ix.N / ix.df[t])
            w = (1.0 + np.log(ix.post_tf[t])) * idf
            sq[d] += w * w
        norms = np.sqrt(sq)
        norms[norms == 0] = 1.0
        self._norms = norms

    def save(self, path: Path) -> None:
        path = Path(path)
        self.index.save(path)
        np.save(path / self.NORMS_FILE, self._norms)

    def load(self, path: Path) -> None:
        path = Path(path)
        self.index = InvertedIndex()
        self.index.load(path)
        f = path / self.NORMS_FILE
        if f.exists():
            self._norms = np.load(f)
        else:
            self._compute_norms()

    # ---- search ----
    def _score(self, query: str):
        ix = self.index
        counts = Counter(t for t in tokenize(query) if t in ix.dictionary)
        if not counts:
            return None
        scores = np.zeros(ix.N)
        cands, qnorm_sq = [], 0.0
        for t, qtf in counts.items():
            idf = math.log(ix.N / ix.df[t])
            wq = (1.0 + math.log(qtf)) * idf
            qnorm_sq += wq * wq
            d = ix.post_docs[t]
            scores[d] += wq * (1.0 + np.log(ix.post_tf[t])) * idf
            cands.append(d)
        if qnorm_sq == 0:
            return None
        scores /= self._norms * math.sqrt(qnorm_sq)
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
        note = (f"TF-IDF cosine {score:.3f}; matched {len(matched)} of {len(terms)} query terms"
                + (f" ({', '.join(matched)})." if matched else "."))
        return {"doc_id": doc_id, "score": score, "matched_terms": matched,
                "components": {"tfidf": score}, "note": note}