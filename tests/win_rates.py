import pandas as pd
from common.config import RESULTS

L = lambda r: pd.read_csv(RESULTS / f"{r}_perquery.csv").set_index("query_id")["NDCG@10"]
b, s, h = L("bm25"), L("semantic"), L("hybrid_minmax_a0.4")
best = pd.concat([b, s], axis=1).max(axis=1)

print(f"hybrid > both:                    {(h > best).mean():.1%}")
print(f"hybrid within 0.01 of the best:   {((h - best).abs() <= 0.01).mean():.1%}")
print(f"hybrid worse than best by > 0.1:  {((best - h) > 0.1).mean():.1%}")
print(f"hybrid > bm25 / < bm25:           {(h > b).mean():.1%} / {(h < b).mean():.1%}")
print(f"hybrid > semantic / < semantic:   {(h > s).mean():.1%} / {(h < s).mean():.1%}")