# pages/4_Comparativa_GSC.py
# GSC comparison page (Streamlit multipage)
# IMPORTANT: Code in English. UI labels/messages in Spanish.

from __future__ import annotations

import math
import re
import unicodedata
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Optional libs (guarded imports)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None


# -------------------------------
# Helpers
# -------------------------------
def normalize_text(s: str) -> str:
    s = str(s)
    s = s.lower()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return " ".join(s.split())


def extract_path(url: str) -> str:
    url = str(url or "")
    # strip protocol, domain, querystring, fragment
    try:
        path = re.sub(r"^[a-z]+://", "", url)
        path = path.split("/", 1)[1] if "/" in path else path
        path = path.split("?", 1)[0]
        path = path.split("#", 1)[0]
    except Exception:
        path = url
    return "/" + path.strip("/")


def compute_similarity(a_list: List[str], b_list: List[str]) -> np.ndarray:
    """
    Similarity between two lists of short texts (names/queries vs pages).
    Prefers TF-IDF cosine; falls back to token Jaccard.
    Returns matrix shape (len(a_list), len(b_list))
    """
    if not a_list or not b_list:
        return np.zeros((len(a_list), len(b_list)))

    # Try TF-IDF
    if TfidfVectorizer is not None and cosine_similarity is not None:
        texts = [normalize_text(x) for x in a_list + b_list]
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        X = vec.fit_transform(texts)
        A = X[: len(a_list)]
        B = X[len(a_list) :]
        return cosine_similarity(A, B)

    # Fallback: token Jaccard (dense matrix)
    def jaccard(s1: str, s2: str) -> float:
        t1 = set(normalize_text(s1).split())
        t2 = set(normalize_text(s2).split())
        if not t1 and not t2:
            return 0.0
        return len(t1 & t2) / max(1, len(t1 | t2))

    m = np.zeros((len(a_list), len(b_list)))
    for i, a in enumerate(a_list):
        for j, b in enumerate(b_list):
            m[i, j] = jaccard(a, b)
    return m


def pick_winner(rows: pd.DataFrame, clicks_col: str, imps_col: str) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype=object)
    if clicks_col in rows.columns and rows[clicks_col].notna().any():
        return rows.sort_values(clicks_col, ascending=False).iloc[0]
    if imps_col in rows.columns and rows[imps_col].notna().any():
        return rows.sort_values(imps_col, ascending=False).iloc[0]
    return rows.iloc[0]


# -------------------------------
# Preconditions
# -------------------------------
st.header("5) Comparativa con Google Search Console")
st.caption("Detecta gaps (sin PLP dedicada) y canibalización (múltiples URLs por concepto) comparando **arquitectura propuesta** vs **estado actual** en GSC.")

# Need taxonomy nodes and GSC data in session
if "taxonomy_nodes" not in st.session_state or not st.session_state["taxonomy_nodes"]:
    st.error("No hay una jerarquía propuesta en sesión. Ve a **3) Clustering y Taxonomía** primero.")
    st.stop()

if "gsc_queries" not in st.session_state and "gsc_pages" not in st.session_state:
    st.warning("No hay datos de GSC cargados. Ve a **1) Carga de datos** y sube el ZIP de GSC (Consultas y Páginas).")
    st.stop()

params = st.session_state.get("params", {})
markets = params.get("markets", list(st.session_state["taxonomy_nodes"].keys()))
available_markets = sorted([mk for mk in st.session_state["taxonomy_nodes"].keys() if mk in markets])

market = st.selectbox("Selecciona mercado para comparar", available_markets, index=0)

nodes = st.session_state["taxonomy_nodes"][market]
root = next((n for n in nodes if n.get("parent_id") is None), None)
children = [n for n in nodes if n.get("parent_id")]

qdf = st.session_state.get("gsc_queries", pd.DataFrame()).copy()
pdf = st.session_state.get("gsc_pages", pd.DataFrame()).copy()

if qdf.empty and pdf.empty:
    st.warning("El ZIP de GSC no trajo ni Consultas ni Páginas. Sube un export válido en **1) Carga de datos**.")
    st.stop()

# Identify columns in GSC exports (robust to locales)
qcol = next((c for c in qdf.columns if "consulta" in c or "query" in c), (qdf.columns[0] if not qdf.empty else "query"))
imp_q = next((c for c in ["impresiones", "impressions"] if c in qdf.columns), None)
clk_q = next((c for c in ["clics", "clicks"] if c in qdf.columns), None)
pos_q = next((c for c in ["posición", "position"] if c in qdf.columns), None)

pcol = next((c for c in pdf.columns if "página" in c or "page" in c or "url" in c), (pdf.columns[0] if not pdf.empty else "page"))
imp_p = next((c for c in ["impresiones", "impressions"] if c in pdf.columns), None)
clk_p = next((c for c in ["clics", "clicks"] if c in pdf.columns), None)
ctr_p = next((c for c in ["ctr"] if c in pdf.columns), None)
pos_p = next((c for c in ["posición", "position"] if c in pdf.columns), None)

# -------------------------------
# Controls
# -------------------------------
with st.expander("Opciones de análisis", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        sim_thr_pages = st.slider("Umbral de similitud (página→nodo)", 0.10, 0.95, 0.35, 0.01,
                                  help="Asignaremos páginas a nodos si superan este umbral.")
    with c2:
        sim_thr_queries = st.slider("Umbral de similitud (query→nodo)", 0.10, 0.95, 0.30, 0.01,
                                    help="Asignaremos consultas a nodos si superan este umbral.")
    with c3:
        min_impressions = st.number_input("Impresiones mín. para considerar", min_value=0, max_value=1_000_000, value=50, step=10)

    st.caption("Sugerencia: empieza con 0.35/0.30 y ajusta según el vocabulario del vertical.")

# -------------------------------
# Prepare data
# -------------------------------
child_names = [c["name"] for c in children]
child_slugs = [c["slug"] for c in children]

# Normalize queries & pages
if not qdf.empty:
    qdf["_query"] = qdf[qcol].astype(str)
    qdf["_qnorm"] = qdf["_query"].map(normalize_text)
    if imp_q:
        qdf = qdf[qdf[imp_q] >= min_impressions] if min_impressions else qdf

if not pdf.empty:
    pdf["_page"] = pdf[pcol].astype(str)
    pdf["_path"] = pdf["_page"].map(extract_path)
    pdf["_pnorm"] = pdf["_path"].map(normalize_text)
    if imp_p:
        pdf = pdf[pdf[imp_p] >= min_impressions] if min_impressions else pdf

# -------------------------------
# Assign queries → nodes
# -------------------------------
queries_assigned = pd.DataFrame()
if not qdf.empty and children:
    sim_q = compute_similarity(qdf["_qnorm"].tolist(), child_names)
    best_idx = np.argmax(sim_q, axis=1)
    best_sim = sim_q[np.arange(sim_q.shape[0]), best_idx]
    queries_assigned = qdf.copy()
    queries_assigned["node_idx"] = best_idx
    queries_assigned["node_name"] = [child_names[i] for i in best_idx]
    queries_assigned["node_slug"] = [child_slugs[i] for i in best_idx]
    queries_assigned["similarity"] = best_sim
    queries_assigned = queries_assigned[queries_assigned["similarity"] >= sim_thr_queries]

# Aggregate demand by node (from queries)
node_demand = pd.DataFrame()
if not queries_assigned.empty:
    agg_cols = [imp_q] if imp_q else []
    node_demand = (
        queries_assigned.groupby(["node_idx", "node_name", "node_slug"], as_index=False)[agg_cols or ["similarity"]]
        .sum()
        .rename(columns={imp_q: "impressions_q"})
    )
else:
    node_demand = pd.DataFrame(columns=["node_idx", "node_name", "node_slug", "impressions_q"])

# -------------------------------
# Assign pages → nodes
# -------------------------------
pages_assigned = pd.DataFrame()
if not pdf.empty and children:
    # Similarity based on path vs node names (also check slug inclusion as boost)
    sim_p = compute_similarity(pdf["_pnorm"].tolist(), child_names)

    # Boost pages where slug substring is present
    for j, slug in enumerate(child_slugs):
        contains = pdf["_path"].str.contains(re.escape(slug), case=False, regex=True)
        sim_p[contains.values, j] += 0.25  # small bonus

    best_idx_p = np.argmax(sim_p, axis=1)
    best_sim_p = sim_p[np.arange(sim_p.shape[0]), best_idx_p]

    pages_assigned = pdf.copy()
    pages_assigned["node_idx"] = best_idx_p
    pages_assigned["node_name"] = [child_names[i] for i in best_idx_p]
    pages_assigned["node_slug"] = [child_slugs[i] for i in best_idx_p]
    pages_assigned["similarity"] = best_sim_p
    pages_assigned = pages_assigned[pages_assigned["similarity"] >= sim_thr_pages]

# Aggregate coverage by node (from pages)
node_coverage = pd.DataFrame()
if not pages_assigned.empty:
    agg = {clk_p: "sum"} if clk_p else {}
    if imp_p:
        agg[imp_p] = "sum"
    node_coverage = (
        pages_assigned.groupby(["node_idx", "node_name", "node_slug"], as_index=False)
        .agg(agg if agg else {"similarity": "count"})
        .rename(columns={clk_p: "clicks_p", imp_p: "impressions_p", "similarity": "pages_count"})
    )
    if "pages_count" not in node_coverage.columns:
        # If we didn't aggregate similarity, compute pages_count explicitly
        node_coverage["pages_count"] = pages_assigned.groupby(["node_idx"]).size().values
else:
    node_coverage = pd.DataFrame(columns=["node_idx", "node_name", "node_slug", "clicks_p", "impressions_p", "pages_count"])

# -------------------------------
# Join demand vs coverage per node
# -------------------------------
nodes_df = pd.DataFrame({
    "node_idx": list(range(len(children))),
    "node_name": child_names,
    "node_slug": child_slugs
})

node_join = nodes_df.merge(node_demand, on=["node_idx","node_name","node_slug"], how="left") \
                    .merge(node_coverage, on=["node_idx","node_name","node_slug"], how="left")

for col in ["impressions_q","clicks_p","impressions_p","pages_count"]:
    if col in node_join.columns:
        node_join[col] = node_join[col].fillna(0).astype(int)

# Heuristic status per node
def status_for_row(r) -> str:
    if r.get("impressions_q", 0) > 0 and r.get("pages_count", 0) == 0:
        return "GAP (sin PLP dedicada)"
    if r.get("pages_count", 0) >= 2:
        return "Canibalización (múltiples URLs)"
    if r.get("impressions_q", 0) == 0 and r.get("pages_count", 0) == 0:
        return "Sin demanda ni cobertura (revisar)"
    return "OK"

node_join["status"] = node_join.apply(status_for_row, axis=1)

# Priority score (normalize impressions_q and inverse of pages_count when gap)
if "impressions_q" in node_join.columns:
    max_imp = max(1, node_join["impressions_q"].max())
    node_join["priority"] = node_join["impressions_q"] / max_imp
    # Boost gaps
    node_join.loc[node_join["status"].str.startswith("GAP"), "priority"] *= 1.2
    # Slight boost to canibalization
    node_join.loc[node_join["status"].str.startswith("Canibalización"), "priority"] *= 1.1
else:
    node_join["priority"] = 0.0

# -------------------------------
# Build query-level GAP table
# -------------------------------
gaps_queries = pd.DataFrame()
if not queries_assigned.empty:
    # Mark as gap if its node has zero pages assigned
    node_pages_map = node_join.set_index("node_idx")["pages_count"].to_dict()
    mask_gap = queries_assigned["node_idx"].map(lambda i: node_pages_map.get(int(i), 0) == 0)
    gaps_queries = queries_assigned[mask_gap].copy()
    gaps_queries["recommended_node"] = gaps_queries["node_name"]
    gaps_queries["suggested_target_url"] = gaps_queries["node_slug"]
    gaps_queries["reason"] = "No dedicated PLP found"
    # If node has multiple pages, mark as cannibalization candidate
    mask_cani = queries_assigned["node_idx"].map(lambda i: node_pages_map.get(int(i), 0) >= 2)
    cani_part = queries_assigned[mask_cani].copy()
    if not cani_part.empty:
        cani_part["recommended_node"] = cani_part["node_name"]
        cani_part["suggested_target_url"] = cani_part["node_slug"]
        cani_part["reason"] = "Potential cannibalization"
        gaps_queries = pd.concat([gaps_queries, cani_part], ignore_index=True)

# -------------------------------
# Build page-level cannibalization table
# -------------------------------
cannibal_pages = pd.DataFrame()
redirects = pd.DataFrame()

if not pages_assigned.empty:
    grp = pages_assigned.groupby(["node_idx", "node_name", "node_slug"])
    rows = []
    redirs = []
    for (idx, name, slug), sub in grp:
        if sub.shape[0] <= 1:
            continue
        # choose a winner (max clicks, else impressions)
        winner = pick_winner(sub, clicks_col=clk_p or "", imps_col=imp_p or "")
        winner_url = winner["_page"]

        # losers
        for _, r in sub.iterrows():
            rows.append({
                "node_name": name,
                "node_slug": slug,
                "page": r["_page"],
                "clicks": int(r.get(clk_p, 0) or 0),
                "impressions": int(r.get(imp_p, 0) or 0),
                "similarity": round(float(r["similarity"]), 3),
                "winner": winner_url,
            })
            if r["_page"] != winner_url:
                redirs.append({
                    "from_url": r["_page"],
                    "to_url": winner_url,
                    "reason": f"Consolidate to best performer for '{name}'"
                })
    cannibal_pages = pd.DataFrame(rows)
    redirects = pd.DataFrame(redirs)

# -------------------------------
# UI: Results
# -------------------------------
st.subheader("Resumen por subcategoría (demanda vs cobertura)")
show_cols = ["node_name","node_slug","impressions_q","pages_count","clicks_p","impressions_p","status","priority"]
show_cols = [c for c in show_cols if c in node_join.columns]
st.dataframe(node_join[show_cols].sort_values("priority", ascending=False), use_container_width=True)
st.download_button(
    "Descargar resumen por subcategoría (CSV)",
    data=node_join.to_csv(index=False).encode("utf-8"),
    file_name=f"gsc_compare_nodes_{market}.csv",
    mime="text/csv"
)

st.subheader("Gaps por consulta (recomendaciones de destino)")
if not gaps_queries.empty:
    cols_order = [c for c in [qcol, imp_q, "node_name", "suggested_target_url", "reason", "similarity"] if c in gaps_queries.columns]
    st.dataframe(gaps_queries[cols_order].sort_values(imp_q if imp_q in gaps_queries.columns else "similarity", ascending=False).head(300),
                 use_container_width=True)
    st.download_button(
        "Descargar gaps por consulta (CSV)",
        data=gaps_queries.to_csv(index=False).encode("utf-8"),
        file_name=f"gsc_gaps_queries_{market}.csv",
        mime="text/csv"
    )
else:
    st.caption("No se detectaron gaps a nivel de consulta con los umbrales actuales.")

st.subheader("Canibalización por páginas (mismo concepto → múltiples URLs)")
if not cannibal_pages.empty:
    st.dataframe(cannibal_pages.sort_values(["node_name","impressions"], ascending=[True, False]).head(500), use_container_width=True)
    st.download_button(
        "Descargar canibalización (CSV)",
        data=cannibal_pages.to_csv(index=False).encode("utf-8"),
        file_name=f"gsc_cannibalization_pages_{market}.csv",
        mime="text/csv"
    )
else:
    st.caption("No se detectaron casos de canibalización con los umbrales actuales.")

st.subheader("Sugerencias de redirects (ganador → consolidación)")
if not redirects.empty:
    st.dataframe(redirects.head(300), use_container_width=True)
    st.download_button(
        "Descargar redirects sugeridos (CSV)",
        data=redirects.to_csv(index=False).encode("utf-8"),
        file_name=f"gsc_redirects_suggested_{market}.csv",
        mime="text/csv"
    )
else:
    st.caption("No se generaron redirects sugeridos.")

st.divider()
st.caption("Consejo: ajusta los umbrales de similitud e impresiones mínimas si los resultados parecen demasiado estrictos o permisivos.")
