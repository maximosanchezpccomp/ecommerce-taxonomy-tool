# src/processing/clustering.py
# High-level keyword clustering utilities (TF-IDF/HDBSCAN/Agglomerative).
# IMPORTANT: Code in English.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Local helpers
from .normalize import ensure_kw_norm
from .embeddings import (
    EmbedderConfig,
    ClusterConfig,
    get_embedder,
    cluster_keywords_with_embeddings,
)

# Optional deps (guarded)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.cluster import AgglomerativeClustering  # type: ignore
    _SKLEARN_OK = True
except Exception:
    TfidfVectorizer = None
    AgglomerativeClustering = None
    _SKLEARN_OK = False

try:
    import hdbscan  # type: ignore
    _HDBSCAN_OK = True
except Exception:
    _HDBSCAN_OK = False


# ---------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------
@dataclass
class ClusteringParams:
    """
    Parameters to control clustering behavior independent from Streamlit UI.
    """
    algorithm: str = "tfidf+agglomerative"  # "embeddings+hdbscan" | "tfidf+agglomerative"
    # Embedding backend (only used if algorithm starts with "embeddings")
    embedder_backend: str = "sbert"         # "sbert" | "openai" | "tfidf"
    embedder_model: str = "all-MiniLM-L6-v2"
    # HDBSCAN/Agglomerative
    min_cluster_size: int = 5
    min_samples: int = 5                    # (HDBSCAN)
    max_items_hint: int = 12                # soft cap for resulting top clusters
    # Selection thresholds
    min_ms_quantile_es: float = 0.60
    min_ms_quantile_default: float = 0.50


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def cluster_keywords(
    df: pd.DataFrame,
    market: str = "es",
    params: Optional[ClusteringParams] = None,
    *,
    openai_client=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cluster a GKP dataframe and select top clusters by market thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        Must include ['keyword','avg_monthly_searches'] (ms optional but recommended).
    market : str
        Market ISO-like code: 'es' | 'pt' | 'fr' | 'it'.
    params : ClusteringParams
        Controls algorithm and thresholds.
    openai_client : object
        Required only if params.embedder_backend == "openai".

    Returns
    -------
    (df_labeled, agg_by_cluster, selected_clusters)
        df_labeled: original df + ['kw_norm','cluster_id','cluster_head']
        agg_by_cluster: ['cluster_id','cluster_head','ms_cluster','count']
        selected_clusters: top clusters after quantile filtering and soft cap
    """
    if df is None or df.empty:
        return (
            pd.DataFrame(columns=["keyword", "avg_monthly_searches", "kw_norm", "cluster_id", "cluster_head"]),
            pd.DataFrame(columns=["cluster_id", "cluster_head", "ms_cluster", "count"]),
            pd.DataFrame(columns=["cluster_id", "cluster_head", "ms_cluster", "count"]),
        )

    p = params or ClusteringParams()
    d = ensure_kw_norm(df)

    if p.algorithm.startswith("embeddings"):
        # Use the generic embedding pipeline
        embed_cfg = EmbedderConfig(
            backend=p.embedder_backend,
            model=p.embedder_model,
            batch_size=64,
            normalize=True,
        )
        # Pick best clustering we can: HDBSCAN if installed else Agglomerative
        algo = "hdbscan" if _HDBSCAN_OK else "agglomerative"
        clus_cfg = ClusterConfig(
            algorithm=algo,
            min_cluster_size=max(2, int(p.min_cluster_size)),
            min_samples=max(1, int(p.min_samples)),
            max_clusters_hint=_heuristic_k(len(d)) if algo == "agglomerative" else None,
        )
        df_labeled, _ = cluster_keywords_with_embeddings(
            d,
            text_col="kw_norm",
            ms_col="avg_monthly_searches",
            embedder=None if p.embedder_backend != "openai" else get_embedder(embed_cfg, openai_client=openai_client),
            embedder_config=embed_cfg if p.embedder_backend != "openai" else None,
            cluster_config=clus_cfg,
        )
    else:
        # Deterministic TF-IDF + Agglomerative fallback
        df_labeled = _cluster_tfidf_agglomerative(d)

    # Aggregate and select
    agg = _aggregate_by_cluster(df_labeled)
    selected = _select_by_market_quantile(
        agg,
        market=market,
        q_es=p.min_ms_quantile_es,
        q_default=p.min_ms_quantile_default,
        max_items=p.max_items_hint,
    )
    return df_labeled, agg, selected


# ---------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------
def _heuristic_k(n: int) -> int:
    """Heuristic number of clusters ~ sqrt(n), at least 2."""
    return max(2, int(math.sqrt(max(1, n))))


def _cluster_tfidf_agglomerative(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight, dependency-minimal clustering using TF-IDF + Agglomerative.
    Adds ['cluster_id','cluster_head'] to the dataframe.
    """
    d = df.copy()
    if d.shape[0] < 8 or not _SKLEARN_OK:
        # Tiny dataset or missing sklearn → single cluster
        d["cluster_id"] = 0
        head = (d.sort_values("avg_monthly_searches", ascending=False)["keyword"].head(1).tolist() or ["General"])[0]
        d["cluster_head"] = head
        return d

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(d["kw_norm"].astype(str).tolist())
    k = _heuristic_k(X.shape[0])

    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = model.fit_predict(X.toarray())
    d["cluster_id"] = labels

    # Head term per cluster by monthly searches
    heads: Dict[int, str] = {}
    d["_ms"] = pd.to_numeric(d.get("avg_monthly_searches", 0), errors="coerce").fillna(0)
    for cid, sub in d.groupby("cluster_id"):
        sub_sorted = sub.sort_values("_ms", ascending=False)
        heads[int(cid)] = sub_sorted.iloc[0]["keyword"] if not sub_sorted.empty else f"Cluster {cid}"
    d["cluster_head"] = d["cluster_id"].map(heads).astype(str)

    d.drop(columns=["_ms"], inplace=True)
    return d


def _aggregate_by_cluster(df_labeled: pd.DataFrame) -> pd.DataFrame:
    """
    Sum monthly searches by cluster and count keywords.
    Returns columns: ['cluster_id','cluster_head','ms_cluster','count']
    """
    ms = pd.to_numeric(df_labeled.get("avg_monthly_searches", 0), errors="coerce").fillna(0).astype(int)
    g = (
        df_labeled.assign(_ms=ms)
        .groupby(["cluster_id", "cluster_head"], as_index=False)["_ms"]
        .sum()
        .rename(columns={"_ms": "ms_cluster"})
    )
    cnt = df_labeled.groupby(["cluster_id"]).size().reset_index(name="count")
    agg = g.merge(cnt, on="cluster_id", how="left").sort_values("ms_cluster", ascending=False)
    return agg.reset_index(drop=True)


def _select_by_market_quantile(
    agg: pd.DataFrame,
    *,
    market: str,
    q_es: float,
    q_default: float,
    max_items: int,
) -> pd.DataFrame:
    """
    Select top clusters using a per-market quantile threshold + a soft cap.
    """
    if agg is None or agg.empty:
        return agg

    q = float(q_es) if (market or "").lower() == "es" else float(q_default)
    thr = agg["ms_cluster"].quantile(q) if 0.0 < q < 1.0 else 0.0
    sel = agg[agg["ms_cluster"] >= thr].sort_values("ms_cluster", ascending=False)
    return sel.head(int(max_items))
