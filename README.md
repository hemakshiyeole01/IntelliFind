# Hybrid Product Search (ESCI small_version)

Shared contract: see Hybrid_Search_Team_Contract.pdf. Everyone imports from `common/`.

## Setup
    pip install -r requirements.txt
    # put the ESCI parquet files in data/esci/
    python -m tests.test_smoke          # offline sanity check

## Semantic
    python -m scripts.build_semantic     # GPU recommended, run once, resumable
    python -m scripts.run_eval semantic  # writes results/semantic.json

Share artifacts/semantic/ (emb.npy + doc_ids.npy) with the team; row i of emb.npy = doc_ids[i].

## For teammates
- Anshupriya: retrievers/tfidf.py, retrievers/bm25.py, retrievers/index.py
- Chhav: retrievers/hybrid.py, retrievers/expansion.py. Semantic also offers
  `embed_query(q)` and `search_by_vector(vec, k)` for Rocchio.
