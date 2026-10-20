"""Offline smoke test (no dataset, no model download). Run: python -m tests.test_smoke"""
import tempfile
from pathlib import Path
import numpy as np

from common.text import tokenize
from common import metrics as M
from retrievers.semantic import SemanticRetriever


class StubModel:
    """Hashed bag-of-words encoder so the test needs no internet."""
    def encode(self, texts, batch_size=32, normalize_embeddings=True,
               convert_to_numpy=True, show_progress_bar=False):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                out[i, hash(tok) % 64] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True); n[n == 0] = 1
        return out / n


DOCS = {"B01": "nike air max 270 running shoes", "B02": "samsung galaxy s23 phone case",
        "B03": "stainless steel cooking pan", "B04": "comfortable walking shoes for standing"}


def test_tokenize():
    assert tokenize("Nike Air Max 270 for the iPhone 15 128GB") == \
        ["nike", "air", "max", "270", "iphone", "15", "128gb"]


def test_metrics():
    assert M.reciprocal_rank(["x", "a"], {"a"}) == 0.5
    assert abs(M.ndcg_at_k(["a", "b"], {"a": 3, "b": 2}, 10) - 1.0) < 1e-9


def test_semantic_roundtrip():
    r = SemanticRetriever(model=StubModel()); r.build(DOCS)
    hits = r.search("samsung s23 case", 3)
    assert hits[0][0] == "B02"
    assert all(isinstance(d, str) for d, _ in hits)
    assert [s for _, s in hits] == sorted([s for _, s in hits], reverse=True)
    assert r.search("   ", 5) == [] and len(r.search("shoes", 100)) == len(DOCS)
    with tempfile.TemporaryDirectory() as tmp:
        r.save(Path(tmp))
        r2 = SemanticRetriever(model=StubModel()); r2.load(Path(tmp))   # no build()
        assert r2.search("samsung s23 case", 3) == hits
        e = r2.explain("samsung s23 case", "B02")
        assert set(e) >= {"doc_id", "score", "matched_terms", "components", "note"}
        assert abs(e["score"] - hits[0][1]) < 1e-5


if __name__ == "__main__":
    test_tokenize(); test_metrics(); test_semantic_roundtrip(); print("all smoke tests passed")
