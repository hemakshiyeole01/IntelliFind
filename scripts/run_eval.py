"""Evaluate one saved retriever on the test split and write results/<name>.json.
    python run_eval.py semantic
    python run_eval.py bm25 --limit 500        # quick smoke run
Hybrid is run from Chhav's own script because it needs two loaded retrievers.
"""
import argparse
import importlib

from common.config import ARTIFACTS
from common.data import load_esci, build_qrels
from common.evaluate import evaluate

CLASSES = {
    "tfidf": ("retrievers.tfidf", "TfidfRetriever"),
    "bm25": ("retrievers.bm25", "Bm25Retriever"),
    "semantic": ("retrievers.semantic", "SemanticRetriever"),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=CLASSES)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--limit", type=int, default=None, help="only first N queries (smoke test)")
    a = ap.parse_args()

    mod, cls = CLASSES[a.name]
    r = getattr(importlib.import_module(mod), cls)()
    r.load(ARTIFACTS / a.name)

    ex, _ = load_esci()
    queries, qrels = build_qrels(ex, a.split)
    if a.limit:
        queries = dict(list(queries.items())[: a.limit])
    run = a.name if not a.limit else f"{a.name}_smoke"
    print(evaluate(r, queries, qrels, run_name=run, split=a.split))
