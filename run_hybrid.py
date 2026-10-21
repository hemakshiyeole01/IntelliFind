"""Alpha sweep on 1000 train queries, then final test runs. Usage:
   python run_hybrid.py --stage sweep
   python run_hybrid.py --stage test --alpha 0.4 [--runs minmax rrf prf]
   python run_hybrid.py --stage all"""
import argparse, random, time
import pandas as pd

import common.evaluate as ev
from common.config import ARTIFACTS, RESULTS, SEED, REL_MIN_GAIN
from common.data import load_esci, build_docs, build_qrels
from retrievers.bm25 import Bm25Retriever
from retrievers.semantic import SemanticRetriever
from retrievers.hybrid import HybridRetriever

N_SWEEP = 1000
ALPHAS = [round(i / 10, 1) for i in range(11)]
COLS = ["P@10", "Recall@10", "MAP", "MRR", "NDCG@10"]


def components():
    lex = Bm25Retriever(); lex.load(ARTIFACTS / "bm25")
    sem = SemanticRetriever(); sem.load(ARTIFACTS / "semantic")
    return lex, sem


def sweep(ex, lex, sem):
    queries, qrels = build_qrels(ex, "train")
    eligible = sorted(q for q in queries if any(g >= REL_MIN_GAIN for g in qrels[q].values()))
    sample = random.Random(SEED).sample(eligible, min(N_SWEEP, len(eligible)))
    sq = {q: queries[q] for q in sample}

    ev.RESULTS = RESULTS / "sweep_runs"            # keep results/ clean (only final runs there)
    h = HybridRetriever(lex, sem); h.enable_pool_cache()
    rows = []
    for fusion in ("minmax", "rrf"):
        h.fusion = fusion
        for a in (ALPHAS if fusion == "minmax" else [None]):   # alpha is unused by RRF
            if a is not None: h.alpha = a
            s = ev.evaluate(h, sq, qrels, run_name=f"sweep_{fusion}_{a}", split="train")
            rows.append({"alpha": a, "fusion": fusion, **{c: s[c] for c in COLS}})
            print(rows[-1])
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "alpha_sweep.csv", index=False)
    mm = df[df.fusion == "minmax"].sort_values(["NDCG@10", "MAP"], ascending=False)
    best = float(mm.iloc[0]["alpha"])
    print("best alpha on train:", best)
    return best


def with_progress(h, total, every=250):
    """Wraps h.search so a progress line is printed every `every` queries."""
    orig, state = h.search, {"n": 0, "t": time.time()}
    def search(q, k=100):
        state["n"] += 1
        if state["n"] % every == 0:
            el = time.time() - state["t"]
            left = el / state["n"] * (total - state["n"]) / 60
            print(f"  {state['n']}/{total} queries, {el/60:.1f} min elapsed, ~{left:.0f} min left",
                  flush=True)
        return orig(q, k)
    h.search = search
    return h


def final_test(ex, pr, lex, sem, best, runs):
    ev.RESULTS = RESULTS
    queries, qrels = build_qrels(ex, "test")
    total = sum(1 for q in queries
                if any(g >= REL_MIN_GAIN for g in qrels.get(q, {}).values()))
    plan = {
        "minmax": (f"hybrid_minmax_a{best}", dict(alpha=best)),
        "rrf": ("hybrid_rrf", dict(fusion="rrf")),
        "prf": (f"hybrid_minmax_a{best}_prf", dict(alpha=best, expand="prf")),
    }
    for key in runs:
        name, kw = plan[key]
        if key == "prf":
            docs, _ = build_docs(ex, pr)
            kw = dict(kw, docs=docs)
        print("running", name, flush=True)
        h = with_progress(HybridRetriever(lex, sem, **kw), total)
        print(ev.evaluate(h, queries, qrels, run_name=name), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sweep", "test", "all"], default="all")
    ap.add_argument("--alpha", type=float, help="use this alpha for the test stage")
    ap.add_argument("--runs", nargs="+", choices=["minmax", "rrf", "prf"],
                    default=["minmax", "rrf", "prf"],
                    help="which final test runs to execute")
    args = ap.parse_args()
    ex, pr = load_esci()
    lex, sem = components()
    best = args.alpha
    if args.stage in ("sweep", "all"):
        best = sweep(ex, lex, sem)
    if args.stage in ("test", "all"):
        final_test(ex, pr, lex, sem, best if best is not None else 0.5, args.runs)