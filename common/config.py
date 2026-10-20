from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "esci"        # parquet files from amazon-science/esci-data
ARTIFACTS = ROOT / "artifacts"           # saved indexes / embeddings (not in git)
RESULTS = ROOT / "results"               # one json + one per-query csv per run

SEED = 42
LOCALE = "us"                            # English only
SMALL_VERSION = 1                        # full small_version, no sampling

RETRIEVE_K = 100                         # every search() returns up to 100 results
POOL_K = 200                             # hybrid pulls this many from each retriever
EVAL_K = 10                              # P@10, Recall@10, NDCG@10

GAIN = {"E": 3, "S": 2, "C": 1, "I": 0}  # graded gain used for NDCG
REL_MIN_GAIN = 2                         # relevant (binary) = Exact or Substitute

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LEN = 128
BM25_K1, BM25_B = 1.5, 0.75
RRF_K = 60
