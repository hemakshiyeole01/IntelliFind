import random, time
from common.data import load_esci, build_docs, build_qrels
from common.config import SEED
import registry
import retrievers.hybrid as H
from retrievers.hybrid import HybridRetriever

def timed(obj, name, label, acc):
    f = getattr(obj, name)
    def w(*a, **k):
        t = time.time(); r = f(*a, **k)
        acc[label] = acc.get(label, 0) + time.time() - t
        return r
    setattr(obj, name, w)

ex, pr = load_esci()
t = time.time(); docs, _ = build_docs(ex, pr); print(f"build_docs: {time.time()-t:.1f}s")
queries, _ = build_qrels(ex, "test")
qs = random.Random(SEED).sample(sorted(set(queries.values())), 31)

lex = registry.load_base("bm25", demo=False)
sem = registry.load_base("semantic", demo=False)
acc = {}
timed(lex, "search", "bm25.search", acc)
timed(sem, "search", "semantic.search", acc)
timed(sem, "embed_query", "embed_query", acc)
timed(sem, "search_by_vector", "search_by_vector", acc)
timed(H, "expand_query", "expand_query", acc)
timed(H, "get_doc_vectors", "get_doc_vectors", acc)

h = HybridRetriever(lex, sem, alpha=0.4, expand="prf", docs=docs)
t = time.time(); h.search(qs[0], 100); print(f"first query (warm-up): {time.time()-t:.1f}s")
acc.clear()
t = time.time()
for q in qs[1:]:
    h.search(q, 100)
total = time.time() - t
print(f"\nPRF: {1000*total/30:.0f} ms/query over 30 queries")
for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
    print(f"  {k:18s} {1000*v/30:8.1f} ms/query")