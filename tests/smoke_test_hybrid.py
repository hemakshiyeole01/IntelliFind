import time
import registry

q = "nike air max 270"

t = time.time()
lex = registry.load_base("bm25", demo=False)
sem = registry.load_base("semantic", demo=False)
print(f"loaded in {time.time()-t:.1f}s")

h = registry.make_hybrid(lex, sem, demo=False, alpha=0.6, fusion="minmax", expand=None)
res = h.search(q, 5)
print(res)
print(h.explain(q, res[0][0]))

h_rrf = registry.make_hybrid(lex, sem, demo=False, fusion="rrf", expand=None)
print(h_rrf.search(q, 5))

# PRF (Rocchio + expansion). The first call builds a cache, so it can take a minute.
t = time.time()
h_prf = registry.make_hybrid(lex, sem, demo=False, alpha=0.6, fusion="minmax", expand="prf")
print(h_prf.search(q, 5))
print(f"PRF search took {time.time()-t:.1f}s")