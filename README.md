# Hybrid Product Search (Amazon ESCI, small_version)

Compares TF-IDF, BM25, semantic and hybrid retrieval on real e-commerce queries and demos the best one in a Streamlit app.
All shared rules live in `Hybrid_Search_Team_Contract.pdf` (one erratum below).

## Status

| Part | Owner | State |
|---|---|---|
| `common/` (config, tokenizer, loader, metrics, evaluator, base interface) | Hemakshi | done |
| Semantic retriever + embeddings (`artifacts/semantic/`) | Hemakshi | done |
| Inverted index, TF-IDF, BM25 | Anshupriya | done |
| Hybrid, query expansion, Rocchio, alpha sweep | Chhav | in progress |
| Streamlit app + registry | Hemakshi | done (hybrid appears once Chhav's files exist) |

## Data

- Source: [amazon-science/esci-data](https://github.com/amazon-science/esci-data)
- Put both files in `data/esci/`:
  `shopping_queries_dataset_examples.parquet` and `shopping_queries_dataset_products.parquet`
- Filter: `small_version == 1`, `product_locale == "us"` (full, no sampling)
- Verified: 601,354 judged pairs, 29,844 queries, **482,105 products** in the corpus
- Labels: E=3, S=2, C=1, I=0 (NDCG gain). Relevant for P/Recall/MAP/MRR = E or S.

## Setup (VS Code, from the project root)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m tests.test_smoke
```

Always run commands from the project root, not from a subfolder.

## Build and evaluate

```
python -m scripts.build_semantic          # one-time, about 3 hours on CPU, resumable
# bm25 / tfidf: build command from Anshupriya (TODO: add it here)
python -m scripts.run_eval semantic
python -m scripts.run_eval bm25
python -m scripts.run_eval tfidf
```

Add `--limit 200` for a quick smoke run (writes `<method>_smoke.json`; delete it before the final table).
Every method must report the same `n_queries` in its json, or the numbers are not comparable.

## Run the app

```
streamlit run app.py
```

- A method appears in the sidebar only when its artifacts exist in `artifacts/<method>/`.
- `hybrid` appears when `artifacts/bm25/`, `artifacts/semantic/` and `retrievers/hybrid.py` all exist.
- No artifacts, or `ESCI_DEMO=1`: demo mode on 14 toy products.
- First real launch builds `artifacts/meta.pkl` (titles, brands) from the parquet files.
- Tabs: Search (score breakdown, matched terms, brand/colour filters), Compare methods, Results table and alpha sweep.

## Project layout

```
common/        config, text (tokenizer), data, metrics, evaluate, base (Retriever interface)
retrievers/    semantic.py, index.py, bm25.py, tfidf.py, hybrid.py*, expansion.py*   (* Chhav)
scripts/       build_semantic.py, run_eval.py
tests/         test_smoke.py, test_app.py
registry.py    method name -> retriever (used by the app)
app.py         Streamlit UI
data/ artifacts/ results/    gitignored, share via drive
```

## Rules everyone follows

- Subclass `common.base.Retriever`. `search(query, k)` returns `[(doc_id: str, raw_score)]`, sorted high to low.
- Use only `common.text.tokenize`, the `common.data` loaders and `common.evaluate`. Never write your own.
- `explain()` returns `doc_id`, `score`, `matched_terms` (list of str), `components`, `note`.
- `components` keys: `bm25` / `tfidf` / `semantic` for single retrievers; **`lexical`, `semantic`, `final`** for hybrid
  (PDF erratum: Section 11 says `bm25`, Section 9.3 is correct).
- Tune only on train queries, report on test. Seed 42.

## What Chhav needs to deliver

- `retrievers/hybrid.py`: `HybridRetriever(lexical, semantic, alpha=0.5, fusion="minmax" | "rrf", expand=None | "prf")`.
  Pool of `POOL_K = 200` from each retriever, per-query min-max, missing doc = 0.
- `retrievers/expansion.py`: query expansion and Rocchio. `SemanticRetriever` offers
  `embed_query(q)` and `search_by_vector(vec, k)` for the semantic side.
- `results/alpha_sweep.csv` with columns `alpha, fusion, NDCG@10, MAP, MRR, P@10, Recall@10`
  (1000 train queries, alpha 0.0 to 1.0 in steps of 0.1), then final runs on test:
  `results/hybrid_*.json`.

## Results

`results/<run>.json` (summary) and `results/<run>_perquery.csv` per method, plus `alpha_sweep.csv`.
Final table: TF-IDF, BM25, Semantic, Hybrid, Hybrid + expansion/feedback on P@10, Recall@10, MAP, MRR, NDCG@10.

## Known limits

- ESCI has no price or rating columns, so price filters and "under 80k" queries cannot be evaluated; filters are brand and colour only.
- Many relevant products are unjudged, so absolute scores are low. Compare methods, not absolute values.
- One embedding model (all-MiniLM-L6-v2), no fine-tuning, no user study.
