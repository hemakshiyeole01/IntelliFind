import pandas as pd
from .config import DATA_DIR, LOCALE, SMALL_VERSION, GAIN

EX_FILE = "shopping_queries_dataset_examples.parquet"
PR_FILE = "shopping_queries_dataset_products.parquet"


def load_esci() -> tuple[pd.DataFrame, pd.DataFrame]:
    ex = pd.read_parquet(DATA_DIR / EX_FILE)
    pr = pd.read_parquet(DATA_DIR / PR_FILE)
    ex = ex[(ex["small_version"] == SMALL_VERSION) & (ex["product_locale"] == LOCALE)]
    pr = pr[pr["product_locale"] == LOCALE]
    return ex, pr


def doc_text(row) -> str:
    """The ONLY definition of a document's text: title + brand + bullet points."""
    parts = [row.get("product_title"), row.get("product_brand"), row.get("product_bullet_point")]
    return " ".join(str(p) for p in parts if isinstance(p, str) and p.strip())


def build_docs(ex: pd.DataFrame, pr: pd.DataFrame):
    """Corpus = every US product that appears in small_version (train + test).
    Returns docs {doc_id: text}, meta {doc_id: {...}}. doc_ids are sorted strings."""
    ids = set(ex["product_id"].astype(str))
    pr = pr[pr["product_id"].astype(str).isin(ids)].drop_duplicates("product_id")
    pr = pr.assign(product_id=pr["product_id"].astype(str)).sort_values("product_id")
    docs, meta = {}, {}
    for row in pr.to_dict("records"):
        pid = row["product_id"]
        docs[pid] = doc_text(row)
        meta[pid] = {k: row.get(k) for k in
                     ("product_title", "product_brand", "product_color", "product_bullet_point")}
    return docs, meta


def build_qrels(ex: pd.DataFrame, split: str):
    """split = 'train' or 'test'. Returns queries {qid: text}, qrels {qid: {doc_id: gain}}."""
    d = ex[ex["split"] == split]
    queries = {str(q): t for q, t in zip(d["query_id"], d["query"])}
    qrels: dict[str, dict[str, int]] = {}
    for q, p, lab in zip(d["query_id"], d["product_id"], d["esci_label"]):
        qrels.setdefault(str(q), {})[str(p)] = GAIN[lab]
    return queries, qrels
