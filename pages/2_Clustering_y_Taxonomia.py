# pages/2_Clustering_y_Taxonomia.py
# Clustering & Taxonomy page (Streamlit multipage)
# IMPORTANT: Code in English. UI labels/messages in Spanish.

import math
import uuid
import unicodedata
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Optional libs (guarded imports)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    AgglomerativeClustering = None
    cosine_similarity = None

# ---------------------------------------
# Local helpers (fallbacks)
# ---------------------------------------
def normalize_text(s: str) -> str:
    s = str(s)
    s = s.lower()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return " ".join(s.split())

def make_slug(name: str, strip_accents: bool = True, lowercase: bool = True) -> str:
    s = normalize_text(name) if strip_accents else str(name)
    if lowercase:
        s = s.lower()
    s = s.replace(" ", "-")
    s = "".join(ch for ch in s if ch.isalnum() or ch in "-/")
    return s.strip("-")

def gen_id() -> str:
    return str(uuid.uuid4())

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))

# ---------------------------------------
# Default top-level per market (editable elsewhere)
# ---------------------------------------
LOCALIZED_ROOT_DEFAULT = {
    "es": "Tecnología",
    "pt": "Tecnologia",
    "fr": "Technologie",
    "it": "Tecnologia",
}

# ---------------------------------------
# Page UI
# ---------------------------------------
st.header("3) Clustering y Taxonomía")
st.caption("Agrupa keywords por temas, propone jerarquía (2–3 niveles) y valida reglas UX/SEO.")

# Preconditions: need GKP data in session
if "gkp_data" not in st.session_state or not st.session_state["gkp_data"]:
    st.error("No hay datos de GKP en sesión. Ve a **1) Carga de datos** y sube al menos un archivo.")
    st.stop()

# Load params or defaults
params = st.session_state.get("params", {})
thresholds = params.get("thresholds", {})
max_levels = int(thresholds.get("max_levels", 3))
max_items_per_level = int(thresholds.get("max_items_per_level", 12))
min_ms_q_es = float(thresholds.get("min_ms_quantile", {}).get("es", 0.60))
min_ms_q_default = float(thresholds.get("min_ms_quantile", {}).get("default", 0.50))
jac_thr = float(thresholds.get("cannibalization", {}).get("jaccard", 0.60))
cos_thr = float(thresholds.get("cannibalization", {}).get("cosine", 0.85))
algo = params.get("clustering", {}).get("algorithm", "tfidf+agglomerative")

naming = params.get("naming", {})
strip_accents = bool(naming.get("strip_accents", True))
lowercase = bool(naming.get("lowercase", True))

markets = params.get("markets", ["es", "pt", "fr", "it"])

with st.expander("Opciones de ejecución", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        algo_choice = st.selectbox(
            "Algoritmo de clustering",
            ["tfidf+agglomerative", "embeddings+hdbscan (fallback a TF-IDF)", "custom (manual)"],
            index=0 if algo == "tfidf+agglomerative" else (1 if "embeddings" in algo else 2),
            help="Para producción se recomienda embeddings + HDBSCAN; aquí usamos TF-IDF como fallback."
        )
    with c2:
        max_items_ui = st.slider("Máx. subcategorías (nivel 2)", 4, 12, max_items_per_level)
    with c3:
        enforce_quantile = st.checkbox("Aplicar umbral mínimo por cuantil", value=True)

# ---------------------------------------
# Core functions
# ---------------------------------------
def cluster_keywords_tfidf(df: pd.DataFrame, max_items: int) -> pd.DataFrame:
    """
    Deterministic lightweight clustering using TF-IDF + Agglomerative.
    Returns df with ['cluster_id','cluster_head'].
    """
    d = df.copy()
    if d.empty:
        d["cluster_id"] = []
        d["cluster_head"] = []
        return d

    # If tiny dataset or libs missing → single cluster
    if TfidfVectorizer is None or AgglomerativeClustering is None or len(d) < 8:
        d["cluster_id"] = 0
        head = d.sort_values("avg_monthly_searches", ascending=False)["keyword"].head(1).tolist()
        d["cluster_head"] = head[0] if head else "General"
        return d

    texts = d["kw_norm"].tolist()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(texts)

    # Heuristic cluster count
    n_clusters = max(2, min(max_items, int(math.sqrt(len(texts)))))
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = model.fit_predict(X.toarray())
    d["cluster_id"] = labels

    # Head term per cluster by monthly searches
    heads = {}
    for cid in sorted(set(labels)):
        sub = d[d["cluster_id"] == cid].sort_values("avg_monthly_searches", ascending=False)
        heads[cid] = sub["keyword"].iloc[0] if not sub.empty else f"Cluster {cid}"
    d["cluster_head"] = d["cluster_id"].map(heads)
    return d

def compute_cluster_ms(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["cluster_id", "cluster_head"], as_index=False)["avg_monthly_searches"]
        .sum()
        .rename(columns={"avg_monthly_searches": "ms_cluster"})
        .sort_values("ms_cluster", ascending=False)
    )
    return agg

def select_clusters_by_quantile(agg: pd.DataFrame, market: str, max_items: int) -> pd.DataFrame:
    if agg.empty:
        return agg
    q = min_ms_q_es if market == "es" else min_ms_q_default
    thr = agg["ms_cluster"].quantile(q) if q > 0 else 0
    selected = agg[agg["ms_cluster"] >= thr].sort_values("ms_cluster", ascending=False)
    # UX: cap by max_items
    return selected.head(max_items)

def build_nodes(root_name: str, market: str, selected_agg: pd.DataFrame, total_ms: int) -> List[Dict]:
    nodes: List[Dict] = []
    root_id = gen_id()
    root = {
        "id": root_id,
        "parent_id": None,
        "name": root_name,
        "slug": f"/{make_slug(root_name, strip_accents, lowercase)}",
        "intent": "mixed",
        "ms": int(total_ms),
        "score": 1.0,
        "recommended_PLPs": [],
        "filters": [],
        "seo": {"title": "", "h1": "", "meta": "", "faq": []},
    }
    nodes.append(root)

    total_sel_ms = max(1, int(selected_agg["ms_cluster"].sum())) if not selected_agg.empty else 1
    for _, r in selected_agg.iterrows():
        name = str(r["cluster_head"]).strip().capitalize()
        ms = int(r["ms_cluster"])
        score = round(ms / total_sel_ms, 3)
        nodes.append({
            "id": gen_id(),
            "parent_id": root_id,
            "name": name,
            "slug": f"/{make_slug(root_name, strip_accents, lowercase)}/{make_slug(name, strip_accents, lowercase)}",
            "intent": "transactional",
            "ms": ms,
            "score": score,
            "recommended_PLPs": [],
            "filters": [],  # se rellenan en la página de Filtros
            "seo": {"title": "", "h1": "", "meta": "", "faq": []},
        })
    return nodes

def detect_cannibalization(nodes: List[Dict]) -> pd.DataFrame:
    """
    Simple overlap check using Jaccard on token sets and TF-IDF cosine on names.
    Marks pairs above thresholds defined in params.
    """
    children = [n for n in nodes if n.get("parent_id")]
    if len(children) < 2:
        return pd.DataFrame(columns=["node_a","node_b","jaccard","cosine","flag"])

    # Tokenize names
    tokens = [normalize_text(n["name"]).split() for n in children]
    names = [n["name"] for n in children]

    # Jaccard
    jacc = np.zeros((len(children), len(children)))
    for i in range(len(children)):
        for j in range(i+1, len(children)):
            jv = jaccard(tokens[i], tokens[j])
            jacc[i, j] = jacc[j, i] = jv

    # Cosine on TF-IDF of names (fallback to zeros if lib missing)
    if TfidfVectorizer is not None and cosine_similarity is not None:
        vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        X = vec.fit_transform([normalize_text(x) for x in names])
        cos = cosine_similarity(X)
    else:
        cos = np.zeros((len(children), len(children)))

    rows = []
    for i in range(len(children)):
        for j in range(i+1, len(children)):
            flag = (jacc[i, j] >= jac_thr) or (cos[i, j] >= cos_thr)
            rows.append({
                "node_a": children[i]["name"],
                "node_b": children[j]["name"],
                "jaccard": round(float(jacc[i, j]), 3),
                "cosine": round(float(cos[i, j]), 3),
                "flag": bool(flag),
            })
    df = pd.DataFrame(rows).sort_values(["flag","jaccard","cosine"], ascending=[False, False, False])
    return df

# ---------------------------------------
# Execution
# ---------------------------------------
gkp_by_market: Dict[str, pd.DataFrame] = st.session_state["gkp_data"]

# Normalize (ensure kw_norm exists)
for mk, df in gkp_by_market.items():
    if "kw_norm" not in df.columns:
        gkp_by_market[mk] = df.assign(kw_norm=df["keyword"].map(normalize_text))

run = st.button("Re-agrupar (ejecutar clustering por mercado)")

if run:
    with st.spinner("Clusterizando y proponiendo jerarquía..."):
        taxonomy_nodes: Dict[str, List[Dict]] = {}
        clusters_preview: Dict[str, pd.DataFrame] = {}
        cannibal_pairs: Dict[str, pd.DataFrame] = {}

        for mk, df in gkp_by_market.items():
            # Skip markets not selected in params
            if markets and mk not in markets:
                continue

            # 1) Clustering
            dfc = df.copy()
            if "tfidf" in algo_choice:
                dfc = cluster_keywords_tfidf(dfc, max_items=max_items_ui)
            else:
                # embeddings/HDBSCAN not implemented in this MVP → fallback
                dfc = cluster_keywords_tfidf(dfc, max_items=max_items_ui)

            # 2) Aggregate MS by cluster and select by thresholds
            agg = compute_cluster_ms(dfc)
            if enforce_quantile:
                selected = select_clusters_by_quantile(agg, market=mk, max_items=max_items_ui)
            else:
                selected = agg.head(max_items_ui)

            # 3) Nodes (root + selected children)
            root_name = LOCALIZED_ROOT_DEFAULT.get(mk, "Tecnología")
            nodes = build_nodes(root_name, mk, selected, total_ms=int(df["avg_monthly_searches"].sum()))

            # 4) Cannibalization detection between child names
            can_df = detect_cannibalization(nodes)

            taxonomy_nodes[mk] = nodes
            clusters_preview[mk] = dfc
            cannibal_pairs[mk] = can_df

        # Save to session
        st.session_state["taxonomy_nodes"] = taxonomy_nodes
        st.session_state["clusters_preview"] = clusters_preview
        st.session_state["cannibal_pairs"] = cannibal_pairs

    st.success("Clustering completado y jerarquía propuesta generada.")

# ---------------------------------------
# Visualization & tables
# ---------------------------------------
if "taxonomy_nodes" in st.session_state and st.session_state["taxonomy_nodes"]:
    st.subheader("Jerarquía propuesta (por mercado)")
    for mk, nodes in st.session_state["taxonomy_nodes"].items():
        st.markdown(f"**{mk.upper()}**")
        df_nodes = pd.DataFrame([
            {"id": n["id"], "parent_id": n["parent_id"] or "", "name": n["name"], "slug": n["slug"], "ms": n["ms"], "score": n["score"]}
            for n in nodes
        ])

        # Treemap (if Plotly available via main app)
        try:
            import plotly.express as px  # local import to avoid hard dependency
            fig = px.treemap(df_nodes, path=["parent_id", "name"], values="ms", hover_data=["slug","score"])
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("No se pudo renderizar el treemap; se muestra tabla.")
            st.dataframe(df_nodes, use_container_width=True)

        st.dataframe(df_nodes.sort_values("score", ascending=False), use_container_width=True)

    st.subheader("Clusters y asignaciones (preview)")
    for mk, dfc in st.session_state.get("clusters_preview", {}).items():
        st.markdown(f"**{mk.upper()}** — {len(dfc):,} keywords")
        preview_cols = [c for c in ["keyword","avg_monthly_searches","kw_norm","cluster_id","cluster_head"] if c in dfc.columns]
        st.dataframe(dfc[preview_cols].sort_values("avg_monthly_searches", ascending=False).head(200), use_container_width=True)

    st.subheader("Posible canibalización entre subcategorías")
    for mk, can_df in st.session_state.get("cannibal_pairs", {}).items():
        if can_df.empty or not can_df["flag"].any():
            st.markdown(f"**{mk.upper()}**: sin señales relevantes de canibalización.")
        else:
            st.markdown(f"**{mk.upper()}** — pares marcados (Jaccard ≥ {jac_thr} o Cosine ≥ {cos_thr})")
            st.dataframe(can_df[can_df["flag"]].head(100), use_container_width=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Guardar resultados en sesión"):
            st.success("Taxonomía y clusters guardados. Continúa con **3) Filtros y Metadatos** o **5) Exportar**.")
    with col2:
        if st.button("Limpiar resultados"):
            for k in ["taxonomy_nodes", "clusters_preview", "cannibal_pairs"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.experimental_rerun()
else:
    st.info("Ejecuta **Re-agrupar** para generar la taxonomía propuesta.")
