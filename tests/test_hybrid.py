from common.base import Retriever
from retrievers.hybrid import HybridRetriever
from retrievers.expansion import expand_query


class Fake(Retriever):
    def __init__(self, name, results): self.name, self.results = name, results
    def build(self, docs): pass
    def search(self, query, k=100): return self.results[:k] if query.strip() else []
    def save(self, path): pass
    def load(self, path): pass


LEX = [("a", 10.0), ("b", 5.0), ("c", 1.0)]
SEM = [("c", 0.9), ("a", 0.5), ("d", 0.1)]

def test_fusion():
    h = HybridRetriever(Fake("bm25", LEX), Fake("semantic", SEM), alpha=0.0)
    assert [d for d, _ in h.search("x")] == ["a", "b", "c", "d"]       # pure lexical
    h.alpha = 1.0; h._cache.clear()
    assert [d for d, _ in h.search("x", 2)] == ["c", "a"]              # pure semantic
    r = HybridRetriever(Fake("bm25", LEX), Fake("semantic", SEM), fusion="rrf").search("x")
    assert {d for d, _ in r} == {"a", "b", "c", "d"}
    assert [s for _, s in r] == sorted([s for _, s in r], reverse=True)

def test_contract():
    h = HybridRetriever(Fake("bm25", LEX), Fake("semantic", SEM), alpha=0.5)
    assert h.search("") == [] and h.search("   ") == []
    e = h.explain("x", "a")
    assert set(e["components"]) == {"lexical", "semantic", "final"} and isinstance(e["note"], str)
    assert h.explain("x", "zzz")["score"] == 0.0                       # doc outside the pool

def test_expand():
    docs = {"d1": "nike air max running shoes mesh", "d2": "nike running shoes lightweight mesh",
            "d3": "leather boots", "d4": "wool socks", "d5": "nike hoodie"}
    q = expand_query("running shoes", None, n_docs=2, n_terms=2, docs=docs, top_doc_ids=["d1", "d2"])
    assert q.startswith("running shoes") and "mesh" in q and "nike" in q

if __name__ == "__main__":
    test_fusion(); test_contract(); test_expand(); print("all hybrid tests passed")