import registry
from common.data import load_esci, build_docs, build_qrels

CASES = {
    "51660": "BM25 wins:       hp 201a black",
    "63917": "Semantic wins:   low calorie cookbooks",
    "51260": "Hybrid wins:     hoover vacuum cleaner",
    "63190": "Failure (0.0):   loccitane hand lotion",
    "44023": "Failure (low):   galaxy s8 plus bodyguards",
}
LABEL = {3: "E", 2: "S", 1: "C", 0: "I"}      # '-' = product not judged for this query

ex, pr = load_esci()
docs, meta = build_docs(ex, pr)
queries, qrels = build_qrels(ex, "test")

lex = registry.load_base("bm25", demo=False)
sem = registry.load_base("semantic", demo=False)
hyb = registry.make_hybrid(lex, sem, demo=False, alpha=0.4, fusion="minmax", expand=None)


def show(name, results, gains, n=5):
    print(f"  [{name}]")
    for rank, (d, s) in enumerate(results[:n], start=1):
        title = (meta[d]["product_title"] or "")[:75]
        print(f"    {rank}. [{LABEL.get(gains.get(d), '-')}] {title}  ({s:.3f})")


for qid, tag in CASES.items():
    q, gains = queries[qid], qrels[qid]
    n_rel = sum(1 for g in gains.values() if g >= 2)
    print(f"\n{'=' * 90}\n{tag}   | query: '{q}'   | judged E/S products: {n_rel}")
    show("BM25", lex.search(q, 5), gains)
    show("SEMANTIC", sem.search(q, 5), gains)
    hres = hyb.search(q, 5)
    show("HYBRID", hres, gains)
    e = hyb.explain(q, hres[0][0])
    print(f"  hybrid top result -> matched terms: {e['matched_terms']}")
    print(f"                       components:    { {k: round(v, 2) for k, v in e['components'].items()} }")