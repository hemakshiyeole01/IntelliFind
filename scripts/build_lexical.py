"""Anshupriya: builds the index, saves artifacts, evaluates tfidf and bm25.
Run from the project root:  python build_lexical.py
"""
from common.config import ARTIFACTS
from common.data import build_docs, build_qrels, load_esci
from common.evaluate import evaluate
from retrievers.bm25 import Bm25Retriever
from retrievers.index import InvertedIndex
from retrievers.tfidf import TfidfRetriever

ex, pr = load_esci()
docs, meta = build_docs(ex, pr)
print("corpus size:", len(docs))

ix = InvertedIndex()
ix.build(docs)  # built once, shared by both retrievers

bm25 = Bm25Retriever(index=ix)
bm25.build(docs)
bm25.save(ARTIFACTS / "bm25")

tfidf = TfidfRetriever(index=ix)
tfidf.build(docs)
tfidf.save(ARTIFACTS / "tfidf")
print("saved artifacts")

queries, qrels = build_qrels(ex, "test")
for r in (bm25, tfidf):
    print(evaluate(r, queries, qrels, run_name=r.name))