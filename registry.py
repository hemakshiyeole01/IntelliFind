"""Maps method names to retrievers for the app. The app never imports a retriever directly.

A method shows up in the app only when its artifacts exist, so a teammate's unfinished
part never breaks the UI. If no real artifacts exist (or ESCI_DEMO=1) the app runs in
demo mode on a tiny toy catalogue.
"""
import importlib
import importlib.util
import os
import pickle
from pathlib import Path

from common.base import Retriever
from common.config import ARTIFACTS
from common.text import tokenize

CLASSES = {
    "tfidf": ("retrievers.tfidf", "TfidfRetriever"),
    "bm25": ("retrievers.bm25", "Bm25Retriever"),
    "semantic": ("retrievers.semantic", "SemanticRetriever"),
}
ORDER = ["tfidf", "bm25", "semantic", "hybrid"]


# ---------------------------------------------------------------- availability
def _has_files(path: Path) -> bool:
    return path.is_dir() and any(p.is_file() for p in path.iterdir())   # ignores _chunks/


def _real_available() -> list[str]:
    found = [n for n in CLASSES if _has_files(ARTIFACTS / n)]
    if ("bm25" in found and "semantic" in found
            and importlib.util.find_spec("retrievers.hybrid") is not None):
        found.append("hybrid")
    return [n for n in ORDER if n in found]


def is_demo() -> bool:
    return os.environ.get("ESCI_DEMO") == "1" or not _real_available()


def available_methods(demo: bool) -> list[str]:
    return list(ORDER) if demo else _real_available()


# ---------------------------------------------------------------- loading
def load_base(name: str, demo: bool) -> Retriever:
    if demo:
        return DemoLexical() if name in ("tfidf", "bm25") else DemoSemantic()
    mod, cls = CLASSES[name]
    r = getattr(importlib.import_module(mod), cls)()
    r.load(ARTIFACTS / name)
    return r


def make_hybrid(lexical: Retriever, semantic: Retriever, demo: bool, **kw) -> Retriever:
    """kw: alpha (0..1), fusion ('minmax' | 'rrf'), expand (None | 'prf')."""
    if demo:
        return DemoHybrid(lexical, semantic, **kw)
    from retrievers.hybrid import HybridRetriever
    return HybridRetriever(lexical, semantic, **kw)


def load_meta(demo: bool) -> dict[str, dict]:
    """{doc_id: {product_title, product_brand, product_color, product_bullet_point}}"""
    if demo:
        return DEMO_META
    cache = ARTIFACTS / "meta.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    from common.data import load_esci, build_docs
    _, meta = build_docs(*load_esci())
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(meta))
    return meta


# ---------------------------------------------------------------- demo mode (toy data)
_P = [
    ("B01", "Nike Air Max 270 Men's Running Shoes", "Nike", "Black"),
    ("B02", "Samsung Galaxy S23 Slim Phone Case", "Spigen", "Clear"),
    ("B03", "Samsung Galaxy S22 Shockproof Cover", "Spigen", "Black"),
    ("B04", "Samsung Galaxy S24 Ultra Case", "OtterBox", "Blue"),
    ("B05", "Cushioned Walking Sneakers for Standing All Day Nurses", "Skechers", "Grey"),
    ("B06", "Non-Stick Cooking Pan Set 10 Piece", "Tefal", "Red"),
    ("B07", "Chef Kitchen Knife Set with Block Gift for Cooks", "Wusthof", "Black"),
    ("B08", "Grilling Cookbook Gift for Dads Who Love Cooking", "Penguin", "White"),
    ("B09", "iPhone 15 128GB Silicone Case", "Apple", "Blue"),
    ("B10", "iPhone 14 Clear Case", "Apple", "Clear"),
    ("B11", "Laptop 16GB RAM 15.6 inch for Video Editing", "Dell", "Silver"),
    ("B12", "Gaming Laptop RTX Fast Rendering Video Editing", "ASUS", "Black"),
    ("B13", "Memory Foam Insoles Comfort for Standing Work", "Dr Scholls", "White"),
    ("B14", "BBQ Grill Tool Set Gift for Dad", "Weber", "Silver"),
]
DEMO_META = {i: {"product_title": t, "product_brand": b, "product_color": c,
                 "product_bullet_point": ""} for i, t, b, c in _P}
DEMO_DOCS = {i: f"{t} {b}" for i, t, b, c in _P}
_SYN = [{"shoes", "sneakers", "footwear", "trainers"}, {"cooking", "cook", "cooks", "kitchen", "chef", "pan", "grill", "bbq"},
        {"case", "cover"}, {"laptop", "notebook", "rendering"}, {"comfortable", "comfort", "cushioned", "foam"},
        {"dad", "dads", "father"}, {"gift", "present"}]


def _concepts(tokens):
    out = set(tokens)
    for g in _SYN:
        if g & set(tokens):
            out |= g
    return out


class _DemoBase(Retriever):
    def build(self, docs): pass
    def save(self, path): pass
    def load(self, path): pass


class DemoLexical(_DemoBase):
    name = "bm25"

    def search(self, query, k=100):
        q = set(tokenize(query))
        hits = [(i, float(len(q & set(tokenize(t))))) for i, t in DEMO_DOCS.items()]
        return sorted([h for h in hits if h[1] > 0], key=lambda x: -x[1])[:k]

    def explain(self, query, doc_id):
        m = sorted(set(tokenize(query)) & set(tokenize(DEMO_DOCS[doc_id])))
        s = dict(self.search(query, 100)).get(doc_id, 0.0)
        return {"doc_id": doc_id, "score": s, "matched_terms": m, "components": {"bm25": s},
                "note": f"exact term match on: {', '.join(m) or 'nothing'}"}


class DemoSemantic(_DemoBase):
    name = "semantic"

    def search(self, query, k=100):
        q = _concepts(tokenize(query))
        out = []
        for i, t in DEMO_DOCS.items():
            d = _concepts(tokenize(t))
            s = len(q & d) / max(len(q | d), 1)
            if s > 0:
                out.append((i, s))
        return sorted(out, key=lambda x: -x[1])[:k]

    def explain(self, query, doc_id):
        s = dict(self.search(query, 100)).get(doc_id, 0.0)
        return {"doc_id": doc_id, "score": s, "matched_terms": [], "components": {"semantic": s},
                "note": "meaning-based match (demo concepts)"}


class DemoHybrid(_DemoBase):
    name = "hybrid"

    def __init__(self, lexical, semantic, alpha=0.5, fusion="minmax", expand=None):
        self.lex, self.sem, self.alpha, self.fusion, self.expand = lexical, semantic, alpha, fusion, expand

    def _norm(self, hits):
        if not hits:
            return {}
        v = [s for _, s in hits]
        lo, hi = min(v), max(v)
        return {d: ((s - lo) / (hi - lo) if hi > lo else 0.0) for d, s in hits}

    def _parts(self, query):
        if self.expand and self.lex.search(query, 1):
            top = DEMO_DOCS[self.lex.search(query, 1)[0][0]]
            query = query + " " + " ".join(t for t in tokenize(top)[:2] if t not in tokenize(query))
        l, s = self.lex.search(query, 200), self.sem.search(query, 200)
        if self.fusion == "rrf":
            f = {}
            for hits in (l, s):
                for r, (d, _) in enumerate(hits, 1):
                    f[d] = f.get(d, 0.0) + 1 / (60 + r)
            return f, self._norm(l), self._norm(s)
        ln, sn = self._norm(l), self._norm(s)
        f = {d: self.alpha * sn.get(d, 0.0) + (1 - self.alpha) * ln.get(d, 0.0) for d in set(ln) | set(sn)}
        return f, ln, sn

    def search(self, query, k=100):
        f = self._parts(query)[0]
        return sorted(f.items(), key=lambda x: -x[1])[:k]

    def explain(self, query, doc_id):
        f, ln, sn = self._parts(query)
        e = self.lex.explain(query, doc_id)
        return {"doc_id": doc_id, "score": f.get(doc_id, 0.0), "matched_terms": e["matched_terms"],
                "components": {"bm25": ln.get(doc_id, 0.0), "semantic": sn.get(doc_id, 0.0),
                               "final": f.get(doc_id, 0.0)},
                "note": "ranked by lexical + semantic fusion"}
