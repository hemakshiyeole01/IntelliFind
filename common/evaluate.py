import json
import time
import pandas as pd
from .config import RESULTS, EVAL_K, RETRIEVE_K, REL_MIN_GAIN
from . import metrics as M


def evaluate(retriever, queries, qrels, run_name=None, split="test",
             k=EVAL_K, depth=RETRIEVE_K):
    """Runs every query, skips queries with no relevant doc, writes results/<run>.json
    and results/<run>_perquery.csv, returns the summary dict."""
    run = run_name or retriever.name
    rows, t0 = [], time.time()
    for qid, qtext in queries.items():
        gains = qrels.get(qid, {})
        rel = {d for d, g in gains.items() if g >= REL_MIN_GAIN}
        if not rel:
            continue
        ranked = [d for d, _ in retriever.search(qtext, depth)]
        rows.append({
            "query_id": qid, "query": qtext,
            f"P@{k}": M.precision_at_k(ranked, rel, k),
            f"Recall@{k}": M.recall_at_k(ranked, rel, k),
            "AP": M.average_precision(ranked, rel),
            "RR": M.reciprocal_rank(ranked, rel),
            f"NDCG@{k}": M.ndcg_at_k(ranked, gains, k),
        })
    df = pd.DataFrame(rows)
    summary = {
        "method": run, "split": split, "n_queries": len(df),
        f"P@{k}": float(df[f"P@{k}"].mean()), f"Recall@{k}": float(df[f"Recall@{k}"].mean()),
        "MAP": float(df["AP"].mean()), "MRR": float(df["RR"].mean()), f"NDCG@{k}": float(df[f"NDCG@{k}"].mean()),
        "ms_per_query": 1000 * (time.time() - t0) / max(len(df), 1),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{run}.json").write_text(json.dumps(summary, indent=2))
    df.to_csv(RESULTS / f"{run}_perquery.csv", index=False)
    return summary
