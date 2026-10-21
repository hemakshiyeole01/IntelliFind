"""Hybrid retriever: fuses a lexical and a semantic retriever (Chhavi)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common.base import Retriever
from common.config import POOL_K, RETRIEVE_K, RRF_K
from retrievers.expansion import expand_query, get_doc_vectors, rocchio_vector

LEXICAL_COMPONENT_KEY = "lexical"   # per team decision: Section 11 erratum, matches Section 9.3  
PRF_DOCS = 3                     # pseudo-relevant docs
PRF_TERMS = 5                    # expansion terms added to the lexical query
PRF_NONREL = 10                  # lowest-ranked semantic pool docs used as pseudo non-relevant


def _minmax(results: list[tuple[str, float]]) -> dict[str, float]:
    if not results:
        return {}
    scores = [s for _, s in results]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return {d: 0.0 for d, _ in results}
    return {d: (s - lo) / (hi - lo) for d, s in results}


class HybridRetriever(Retriever):
    name = "hybrid"

    def __init__(self, lexical: Retriever, semantic: Retriever, alpha: float = 0.5,
                 fusion: str = "minmax", expand: str | None = None,
                 docs: dict[str, str] | None = None) -> None:
        if expand in ("none", "", False):
            expand = None
        if fusion not in ("minmax", "rrf"):
            raise ValueError("fusion must be 'minmax' or 'rrf'")
        if expand not in (None, "prf"):
            raise ValueError("expand must be None or 'prf'")
        self.lexical, self.semantic = lexical, semantic
        self.alpha, self.fusion, self.expand = float(alpha), fusion, expand
        self.docs = docs                       # only needed for PRF; lazy-loaded if None
        self._cache: dict = {}                 # memo of pools (results never change)
        self._cache_max: int | None = 64

    def enable_pool_cache(self) -> None:
        """Unbounded memo: lets an alpha sweep reuse pools instead of re-searching."""
        self._cache_max = None

    # ------------------------------------------------------------ interface
    def build(self, docs: dict[str, str]) -> None:
        self.docs = docs                       # components are already built

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "hybrid.json").write_text(json.dumps(
            {"alpha": self.alpha, "fusion": self.fusion, "expand": self.expand}))

    def load(self, path: Path) -> None:
        cfg = json.loads((Path(path) / "hybrid.json").read_text())
        self.alpha, self.fusion, self.expand = cfg["alpha"], cfg["fusion"], cfg["expand"]
        self._cache.clear()

    # ------------------------------------------------------------ fusion
    def _fuse_detail(self, lex, sem) -> dict[str, tuple[float, float, float]]:
        """doc_id -> (lexical part, semantic part, final)."""
        out = {}
        if self.fusion == "minmax":
            ln, sn = _minmax(lex), _minmax(sem)
            for d in ln.keys() | sn.keys():
                l, s = ln.get(d, 0.0), sn.get(d, 0.0)
                out[d] = (l, s, self.alpha * s + (1 - self.alpha) * l)
        else:  # rrf: ranks start at 1, lists are already sorted DESC
            lr = {d: r for r, (d, _) in enumerate(lex, start=1)}
            sr = {d: r for r, (d, _) in enumerate(sem, start=1)}
            for d in lr.keys() | sr.keys():
                l = 1.0 / (RRF_K + lr[d]) if d in lr else 0.0
                s = 1.0 / (RRF_K + sr[d]) if d in sr else 0.0
                out[d] = (l, s, l + s)
        return out

    @staticmethod
    def _ranked(detail) -> list[tuple[str, float]]:
        return sorted(((d, v[2]) for d, v in detail.items()), key=lambda x: (-x[1], x[0]))

    # ------------------------------------------------------------ retrieval
    def _retrieve(self, query: str):
        key = (query,) if self.expand is None else (query, self.alpha, self.fusion)
        if key in self._cache:
            return self._cache[key]
        lex = self.lexical.search(query, POOL_K)
        sem = self.semantic.search(query, POOL_K)
        if self.expand == "prf":
            top = [d for d, _ in self._ranked(self._fuse_detail(lex, sem))[:PRF_DOCS]]
            if top:
                # lexical side: expanded text query
                eq = expand_query(query, self.lexical, n_docs=PRF_DOCS, n_terms=PRF_TERMS,
                                  docs=self.docs, top_doc_ids=top)
                lex = self.lexical.search(eq, POOL_K) or lex
                # semantic side: Rocchio on the query vector
                q_vec = np.asarray(self.semantic.embed_query(query), dtype=np.float32).reshape(-1)
                rel = get_doc_vectors(self.semantic, top)
                nonrel_ids = [d for d, _ in sem[-PRF_NONREL:] if d not in top]
                nonrel = get_doc_vectors(self.semantic, nonrel_ids)
                q_new = rocchio_vector(q_vec, rel, nonrel)
                sem = self.semantic.search_by_vector(q_new, POOL_K) or sem
        if self._cache_max is not None and len(self._cache) >= self._cache_max:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = (lex, sem)
        return lex, sem

    def search(self, query: str, k: int = RETRIEVE_K) -> list[tuple[str, float]]:
        if not query or not query.strip():
            return []
        lex, sem = self._retrieve(query)
        return self._ranked(self._fuse_detail(lex, sem))[:k]

    def explain(self, query: str, doc_id: str) -> dict:
        lex, sem = self._retrieve(query)
        l, s, f = self._fuse_detail(lex, sem).get(doc_id, (0.0, 0.0, 0.0))
        try:
            matched = list(self.lexical.explain(query, doc_id).get("matched_terms", []))
        except Exception:
            matched = []
        params = f"alpha={self.alpha}" if self.fusion == "minmax" else "RRF"
        prf = " + PRF" if self.expand else ""
        return {
            "doc_id": doc_id, "score": f, "matched_terms": matched,
            "components": {LEXICAL_COMPONENT_KEY: l, "semantic": s, "final": f},
            "note": f"Hybrid ({self.fusion}{prf}, {params}): lexical {l:.2f}, semantic {s:.2f}, final {f:.2f}.",
        }