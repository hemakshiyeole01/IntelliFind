"""Run: python -m tests.test_anshupriya   (offline, no ESCI data needed)"""
import subprocess
import sys
import tempfile
from pathlib import Path

from retrievers.bm25 import Bm25Retriever
from retrievers.index import InvertedIndex
from retrievers.tfidf import TfidfRetriever

DOCS = {
    "B001": "Samsung Galaxy S23 case clear shockproof",
    "B002": "Comfortable running shoes for standing all day",
    "B003": "Apple iPhone 14 case leather wallet",
    "B004": "Samsung 128GB microSD memory card",
    "B005": "Galaxy S23 screen protector tempered glass",
}


def check(r, label):
    assert r.search("") == [], label
    assert r.search("zzzunknownword") == [], label
    res = r.search("samsung galaxy s23 case", 100)
    assert res and res[0][0] == "B001", (label, res)
    scores = [s for _, s in res]
    assert scores == sorted(scores, reverse=True), label
    assert all(isinstance(d, str) for d, _ in res), label
    assert res == r.search("samsung galaxy s23 case", 100), "not deterministic"
    assert len(r.search("samsung galaxy s23 case", 2)) == 2
    e = r.explain("samsung galaxy s23 case", "B001")
    assert "samsung" in e["matched_terms"] and e["components"], e
    print("ok", label, res[:3])


def main():
    shared = InvertedIndex()
    shared.build(DOCS)
    assert shared.doc_ids == sorted(DOCS)
    for t, p in shared.postings.items():
        assert [d for d, _ in p] == sorted(d for d, _ in p)
        assert shared.df[t] == len(p)

    for cls in (Bm25Retriever, TfidfRetriever):
        r = cls(index=shared)
        r.build(DOCS)
        check(r, cls.name + " (built)")
        with tempfile.TemporaryDirectory() as tmp:
            r.save(Path(tmp) / cls.name)
            # fresh process: load() then search() without build()
            code = (
                "from retrievers.%s import %s as C;"
                "from pathlib import Path;"
                "r=C();r.load(Path(r'%s'));"
                "print(r.search('galaxy s23 case',3))"
            ) % (cls.__module__.split(".")[-1], cls.__name__, Path(tmp) / cls.name)
            out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
            assert out.returncode == 0 and "B001" in out.stdout, out.stderr
            print("ok", cls.name, "fresh-process load")
    print("ALL PASSED")


if __name__ == "__main__":
    main()