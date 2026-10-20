"""Encode the full small_version corpus once and save artifacts/semantic/.
Run on a GPU (Colab/Kaggle). Safe to re-run: finished chunks are reused.
    python build_semantic.py
"""
import shutil
import numpy as np
from common.config import ARTIFACTS, SEED
from common.data import load_esci, build_docs
from retrievers.semantic import SemanticRetriever

if __name__ == "__main__":
    np.random.seed(SEED)
    ex, pr = load_esci()
    docs, _ = build_docs(ex, pr)
    print(f"corpus size: {len(docs)} products")           # tell the team this number
    out = ARTIFACTS / "semantic"
    r = SemanticRetriever()
    r.build(docs, checkpoint_dir=out / "_chunks")
    r.save(out)
    shutil.rmtree(out / "_chunks", ignore_errors=True)
    print("saved", out, r.emb.shape)
