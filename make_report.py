import json
import numpy as np
import pandas as pd
from common.config import RESULTS

RUNS = [("TF-IDF", "tfidf"), ("BM25", "bm25"), ("Semantic", "semantic"),
           ("Hybrid (minmax, a=0.4)", "hybrid_minmax_a0.4"), ("Hybrid (RRF)", "hybrid_rrf"),
           ("Hybrid + PRF", "hybrid_minmax_a0.4_prf")]
COLS = ["P@10", "Recall@10", "MAP", "MRR", "NDCG@10", "n_queries", "ms_per_query"]

rows = []
for label, run in RUNS:
       p = RESULTS / f"{run}.json"
       if not p.exists():
           print("missing:", p); continue
       s = json.loads(p.read_text())
       rows.append({"Method": label, **{c: s[c] for c in COLS}})
tbl = pd.DataFrame(rows)
print(tbl.round(4).to_string(index=False))
tbl.to_csv(RESULTS / "comparison_table.csv", index=False)

def load(run):
       return pd.read_csv(RESULTS / f"{run}_perquery.csv").set_index("query_id")

PQ = {"P@10": "P@10", "Recall@10": "Recall@10", "MAP": "AP", "MRR": "RR", "NDCG@10": "NDCG@10"}

def paired(a, b, n_boot=2000, seed=42):
       """Mean per-query difference (b - a) with a bootstrap 95% CI."""
       A, B = load(a), load(b)
       ids = A.index.intersection(B.index)
       rng = np.random.default_rng(seed)
       print(f"\n{b} minus {a}  (n={len(ids)})")
       for m, col in PQ.items():
           d = (B.loc[ids, col] - A.loc[ids, col]).to_numpy()
           boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
           lo, hi = np.percentile(boots, [2.5, 97.5])
           flag = "significant" if lo > 0 or hi < 0 else "not significant"
           print(f"  {m:10s} {d.mean():+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  {flag}")

for a, b in [("bm25", "hybrid_minmax_a0.4"), ("semantic", "hybrid_minmax_a0.4"),
                ("hybrid_minmax_a0.4", "hybrid_minmax_a0.4_prf"),
                ("hybrid_minmax_a0.4", "hybrid_rrf")]:
       try:
           paired(a, b)
       except FileNotFoundError as e:
           print("skipped:", e)

   # Example queries for the slides
try:
       b, s, h = load("bm25"), load("semantic"), load("hybrid_minmax_a0.4")
       ids = b.index.intersection(s.index).intersection(h.index)
       df = pd.DataFrame({"query": b.loc[ids, "query"], "bm25": b.loc[ids, "NDCG@10"],
                          "semantic": s.loc[ids, "NDCG@10"], "hybrid": h.loc[ids, "NDCG@10"]})
       show = lambda title, x: print(f"\n{title}\n", x.head(8).round(3).to_string())
       show("BM25 beats semantic", df.assign(gap=df.bm25 - df.semantic).sort_values("gap", ascending=False))
       show("Semantic beats BM25", df.assign(gap=df.semantic - df.bm25).sort_values("gap", ascending=False))
       show("Hybrid beats both", df.assign(gap=df.hybrid - df[["bm25", "semantic"]].max(axis=1)).sort_values("gap", ascending=False))
except FileNotFoundError as e:
       print("example queries skipped:", e)