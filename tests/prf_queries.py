import pandas as pd
from common.config import RESULTS

a = pd.read_csv(RESULTS / "hybrid_minmax_a0.4_perquery.csv").set_index("query_id")
b = pd.read_csv(RESULTS / "hybrid_minmax_a0.4_prf_perquery.csv").set_index("query_id")
d = pd.DataFrame({"query": a["query"], "RR_hybrid": a["RR"], "RR_prf": b["RR"],
                  "AP_hybrid": a["AP"], "AP_prf": b["AP"]})
d["dRR"] = d.RR_prf - d.RR_hybrid
print(f"PRF lowers RR on {(d.dRR < 0).mean():.1%} of queries, raises it on {(d.dRR > 0).mean():.1%}")
print("\nPRF hurts most:\n", d.sort_values("dRR").head(8).round(3).to_string())
print("\nPRF helps most:\n", d.sort_values("dRR").tail(8).round(3).to_string())