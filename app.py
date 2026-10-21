"""Streamlit app. Run from the project root:  streamlit run app.py"""
import html
import json
import re

import pandas as pd
import streamlit as st

import registry as R
from common.config import RESULTS, RETRIEVE_K

st.set_page_config(page_title="Hybrid Product Search", layout="wide")
DEMO = R.is_demo()
METHODS = R.available_methods(DEMO)
DEMOS = ["samsung s23 case", "comfortable shoes for standing all day", "gift for dad who likes cooking"]


@st.cache_resource(show_spinner="Loading product data...")
def get_meta(demo):
    return R.load_meta(demo)


@st.cache_resource(show_spinner="Loading retriever...")
def get_base(name, demo):
    return R.load_base(name, demo)


def get_retriever(name, alpha, fusion, expand):
    if name == "hybrid":
        return R.make_hybrid(get_base("bm25", DEMO), get_base("semantic", DEMO), DEMO,
                             alpha=alpha, fusion=fusion, expand=expand)
    return get_base(name, DEMO)


def highlight(text, terms):
    t = html.escape(str(text or ""))
    terms = sorted({x for x in terms if x}, key=len, reverse=True)
    if terms:
        pat = r"(?i)\b(" + "|".join(re.escape(x) for x in terms) + r")\b"
        t = re.sub(pat, r"<mark>\1</mark>", t)
    return t


def fmt(components):
    return "  |  ".join(f"{k} {v:.3f}" for k, v in components.items())


# ------------------------------------------------------------------ sidebar
st.sidebar.title("Settings")
method = st.sidebar.selectbox("Method", METHODS, index=len(METHODS) - 1)
alpha, fusion, expand = 0.5, "minmax", None
if method == "hybrid":
    alpha = st.sidebar.slider("alpha (weight on semantic)", 0.0, 1.0, 0.5, 0.05)
    fusion = st.sidebar.selectbox("Fusion", ["minmax", "rrf"])
    expand = {"none": None, "prf (pseudo-relevance feedback)": "prf"}[
        st.sidebar.selectbox("Query expansion", ["none", "prf (pseudo-relevance feedback)"])]
topk = st.sidebar.slider("Results to show", 5, 30, 10)

st.title("Hybrid Product Search")
if DEMO:
    st.info("Demo mode: toy catalogue of 14 products. Real methods appear automatically "
            "once their artifacts exist in artifacts/.")
else:
    st.caption("Available methods: " + ", ".join(METHODS))

tab_search, tab_compare, tab_results = st.tabs(["Search", "Compare methods", "Results"])
st.session_state.setdefault("q", "")
META = get_meta(DEMO)


def set_q(x):
    st.session_state["q"] = x


with tab_search:
    cols = st.columns(len(DEMOS))
    for c, d in zip(cols, DEMOS):
        c.button(d, on_click=set_q, args=(d,))
    query = st.text_input("Search products", key="q", placeholder="Type a query and press Enter, e.g. Nike Air Max 270")

    if query.strip():
        r = get_retriever(method, alpha, fusion, expand)
        with st.spinner("Searching..."):
            hits = r.search(query, RETRIEVE_K)
        brands = sorted({b for d, _ in hits if isinstance(b := META.get(d, {}).get("product_brand"), str)})
        colors = sorted({c for d, _ in hits if isinstance(c := META.get(d, {}).get("product_color"), str)})
        f1, f2 = st.columns(2)
        pick_b = f1.multiselect("Brand filter", brands)
        pick_c = f2.multiselect("Colour filter", colors)
        shown = 0
        for rank, (did, score) in enumerate(hits, 1):
            m = META.get(did, {})
            if (pick_b and m.get("product_brand") not in pick_b) or \
               (pick_c and m.get("product_color") not in pick_c):
                continue
            ex = r.explain(query, did)
            with st.container(border=True):
                st.markdown(f"**#{rank}** &nbsp; " + highlight(m.get("product_title") or did, ex["matched_terms"]),
                            unsafe_allow_html=True)
                st.caption(f"{m.get('product_brand') or 'unknown brand'}  ·  {m.get('product_color') or 'n/a'}"
                           f"  ·  id {did}  ·  score {score:.3f}")
                if ex["components"]:
                    st.caption("Score breakdown: " + fmt(ex["components"]))
                if ex["note"]:
                    st.caption("Why: " + ex["note"])
            shown += 1
            if shown >= topk:
                break
        if shown == 0:
            st.warning("No results (try fewer filters or a different query).")

with tab_compare:
    st.write("Same query, methods side by side.")
    cmp_methods = [x for x in ("bm25", "semantic", "hybrid") if x in METHODS]
    if not st.session_state["q"].strip():
        st.write("Type a query in the Search tab first.")
    elif len(cmp_methods) < 2:
        st.write("Need at least two methods available to compare.")
    else:
        cols = st.columns(len(cmp_methods))
        for col, name in zip(cols, cmp_methods):
            rr = get_retriever(name, alpha, fusion, expand)
            col.subheader(name + (f" (alpha {alpha})" if name == "hybrid" else ""))
            for rank, (did, score) in enumerate(rr.search(st.session_state["q"], 5), 1):
                m = META.get(did, {})
                col.markdown(f"**{rank}.** {html.escape(str(m.get('product_title') or did))}")
                col.caption(f"{m.get('product_brand') or ''}  ·  score {score:.3f}")

with tab_results:
    rows = []
    for p in sorted(RESULTS.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except ValueError:
            continue
        if isinstance(d, dict) and "method" in d:
            rows.append(d)
    if not rows:
        st.write("No result files yet. Run `python run_eval.py <method>` to create results/<method>.json.")
    else:
        df = pd.DataFrame(rows).sort_values("NDCG@10", ascending=False)
        score_cols = [c for c in ["P@10", "Recall@10", "MAP", "MRR", "NDCG@10"] if c in df]
        fmt_map = {c: "{:.4f}" for c in score_cols}
        fmt_map["ms_per_query"] = "{:.1f}"
        st.dataframe(df.style.highlight_max(
            subset=score_cols, props="background-color:#166534;color:#ffffff;font-weight:bold"
        ).format({k: v for k, v in fmt_map.items() if k in df}), hide_index=True)
    sweep = RESULTS / "alpha_sweep.csv"
    if sweep.exists():
        sw = pd.read_csv(sweep)
        st.subheader("Alpha sweep")
        st.line_chart(sw.pivot_table(index="alpha", columns="fusion", values="NDCG@10"))
