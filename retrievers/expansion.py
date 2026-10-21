"""Query expansion + Rocchio feedback (Chhavi)."""
from __future__ import annotations

import math
import pickle
from collections import Counter

import numpy as np

from common.config import ARTIFACTS
from common.text import tokenize

MIN_DF = 2          # ignore terms seen in fewer than 2 catalogue docs (typos, unique ids)
_CACHE: dict = {}


def _load_docs() -> dict[str, str]:
    """Fallback when no docs dict is passed: load the shared corpus once."""
    if "docs" not in _CACHE:
        from common.data import load_esci, build_docs
        ex, pr = load_esci()
        _CACHE["docs"], _ = build_docs(ex, pr)
    return _CACHE["docs"]


def _get_df(docs: dict[str, str]) -> tuple[dict[str, int], int]:
    """Document frequency of every term in the corpus (cached in memory and on disk)."""
    key = id(docs)
    if key in _CACHE.setdefault("df", {}):
        return _CACHE["df"][key]
    path = ARTIFACTS / "expansion" / "df.pkl"
    persist = docs is _CACHE.get("docs")          # only cache the real corpus on disk
    if persist and path.exists():
        n, df = pickle.loads(path.read_bytes())
        if n == len(docs):
            _CACHE["df"][key] = (df, n)
            return df, n
    df: Counter = Counter()
    for text in docs.values():
        df.update(set(tokenize(text)))
    if persist:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps((len(docs), dict(df))))
    _CACHE["df"][key] = (df, len(docs))
    return df, len(docs)


def expand_query(query: str, retriever, n_docs: int = 3, n_terms: int = 5,
                 docs: dict[str, str] | None = None,
                 top_doc_ids: list[str] | None = None) -> str:
    """Append the n_terms highest-weight terms from the top n_docs results.

    Weight of a term = sum of (1 + log tf) * idf over the feedback docs, multiplied by
    the number of feedback docs containing it (so one-doc oddities do not win).
    `top_doc_ids` lets the caller supply the feedback docs (HybridRetriever does).
    """
    if docs is None:
        docs = _load_docs()
    if top_doc_ids is None:
        top_doc_ids = [d for d, _ in retriever.search(query, n_docs)]
    top_doc_ids = top_doc_ids[:n_docs]
    df, n = _get_df(docs)
    q_terms = set(tokenize(query))

    weight: Counter = Counter()
    seen_in: Counter = Counter()
    for did in top_doc_ids:
        for t, c in Counter(tokenize(docs.get(did, ""))).items():
            if t in q_terms or df.get(t, 0) < MIN_DF:
                continue
            weight[t] += (1 + math.log(c)) * math.log(n / df[t])
            seen_in[t] += 1
    scored = {t: w * seen_in[t] for t, w in weight.items()}
    best = sorted(scored, key=lambda t: (-scored[t], t))[:n_terms]   # deterministic ties
    return " ".join([query] + best).strip()


# ------------------------------------------------------------------ Rocchio (vector space)
def rocchio_vector(q_vec: np.ndarray, rel_vecs: np.ndarray, nonrel_vecs: np.ndarray,
                   alpha: float = 1.0, beta: float = 0.75, gamma: float = 0.15) -> np.ndarray:
    """q_new = alpha*q + beta*mean(relevant) - gamma*mean(non-relevant), then L2-normalised."""
    new = alpha * np.asarray(q_vec, dtype=np.float32)
    if len(rel_vecs):
        new = new + beta * rel_vecs.mean(axis=0)
    if len(nonrel_vecs):
        new = new - gamma * nonrel_vecs.mean(axis=0)
    norm = np.linalg.norm(new)
    return (new / norm).astype(np.float32) if norm > 0 else np.asarray(q_vec, dtype=np.float32)


_STORE: dict = {}

def _semantic_matrix(semantic):
    """(embedding matrix, {doc_id: row}). Uses the retriever's attributes if present,
    otherwise the contract files artifacts/semantic/emb.npy + doc_ids.npy."""
    if "emb" not in _STORE:
        emb = getattr(semantic, "emb", None)
        ids = getattr(semantic, "doc_ids", None)
        if emb is None or ids is None:
            base = ARTIFACTS / "semantic"
            emb = np.load(base / "emb.npy", mmap_mode="r")
            ids = np.load(base / "doc_ids.npy", allow_pickle=True)
        _STORE["emb"] = emb
        _STORE["rows"] = {str(d): i for i, d in enumerate(ids)}
    return _STORE["emb"], _STORE["rows"]


def get_doc_vectors(semantic, doc_ids: list[str]) -> np.ndarray:
    emb, rows = _semantic_matrix(semantic)
    idx = [rows[d] for d in doc_ids if d in rows]
    if not idx:
        return np.zeros((0, emb.shape[1]), dtype=np.float32)
    return np.asarray(emb[idx], dtype=np.float32)