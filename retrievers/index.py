"""Hand-built inverted index (no sklearn / Whoosh). Shared by tfidf and bm25."""
from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from common.text import tokenize


class InvertedIndex:
    """Contract fields (these are what gets pickled):
    doc_ids   : sorted list[str], same order as the semantic embeddings
    dictionary: term -> term_id
    postings  : term -> [(doc_index, tf)] sorted by doc_index
    df        : term -> document frequency
    doc_len   : token count per doc (after stopword removal)
    Derived (rebuilt by finalize(), never pickled): N, avglen, numpy views.
    """

    FILE = "index.pkl"

    def __init__(self) -> None:
        self.doc_ids: list[str] = []
        self.dictionary: dict[str, int] = {}
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.df: dict[str, int] = {}
        self.doc_len: list[int] = []
        self.N = 0
        self.avglen = 1.0
        self.doc_len_arr = np.zeros(0)
        self.post_docs: dict[str, np.ndarray] = {}
        self.post_tf: dict[str, np.ndarray] = {}
        self.id2idx: dict[str, int] = {}

    # ---- build / derive ----
    def build(self, docs: dict[str, str]) -> None:
        self.doc_ids = sorted(docs)
        postings: dict[str, list[tuple[int, int]]] = {}
        doc_len: list[int] = []
        for di, did in enumerate(self.doc_ids):
            toks = tokenize(docs[did])
            doc_len.append(len(toks))
            for term, tf in Counter(toks).items():
                postings.setdefault(term, []).append((di, tf))  # di ascending => sorted
        self.postings = postings
        self.dictionary = {t: i for i, t in enumerate(sorted(postings))}
        self.df = {t: len(p) for t, p in postings.items()}
        self.doc_len = doc_len
        self.finalize()

    def finalize(self) -> None:
        self.N = len(self.doc_ids)
        self.doc_len_arr = np.asarray(self.doc_len, dtype=np.float64)
        self.avglen = float(self.doc_len_arr.mean()) if self.N and self.doc_len_arr.sum() > 0 else 1.0
        self.id2idx = {d: i for i, d in enumerate(self.doc_ids)}
        self.post_docs, self.post_tf = {}, {}
        for t, plist in self.postings.items():
            arr = np.asarray(plist, dtype=np.int64)
            self.post_docs[t] = arr[:, 0]
            self.post_tf[t] = arr[:, 1].astype(np.float64)

    # ---- persistence ----
    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state = {k: getattr(self, k) for k in ("doc_ids", "dictionary", "postings", "df", "doc_len")}
        with open(path / self.FILE, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: Path) -> None:
        with open(Path(path) / self.FILE, "rb") as f:
            state = pickle.load(f)
        for k, v in state.items():
            setattr(self, k, v)
        self.finalize()

    # ---- helpers used by the retrievers ----
    def query_terms(self, query: str) -> list[str]:
        """Unique known query terms, in query order. Unknown terms are ignored."""
        seen, out = set(), []
        for t in tokenize(query):
            if t in self.dictionary and t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def tf_of(self, term: str, doc_index: int) -> int:
        docs = self.post_docs.get(term)
        if docs is None:
            return 0
        pos = int(np.searchsorted(docs, doc_index))
        if pos < len(docs) and docs[pos] == doc_index:
            return int(self.post_tf[term][pos])
        return 0


def top_k(index: InvertedIndex, scores: np.ndarray, cand: np.ndarray, k: int) -> list[tuple[str, float]]:
    """Top-k over candidate docs only. Ties broken by doc index => deterministic."""
    if k <= 0 or cand.size == 0:
        return []
    s = scores[cand]
    if cand.size > k:
        keep = np.argpartition(-s, k - 1)[:k]
        cand, s = cand[keep], s[keep]
    order = np.lexsort((cand, -s))  # score desc, then doc index asc
    return [(index.doc_ids[int(cand[i])], float(s[i])) for i in order]