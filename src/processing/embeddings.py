# src/processing/embeddings.py
# Embedding utilities + clustering (HDBSCAN/Agglomerative) for keyword grouping.
# IMPORTANT: Code in English.

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional deps (guarded)
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _SBERT_AVAILABLE = True
except Exception:
    _SBERT_AVAILABLE = False

try:
    import hdbscan  # type: ignore
    _HDBSCAN_AVAILABLE = True
except Exception:
    _HDBSCAN_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.cluster import AgglomerativeClustering  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    _SKLEARN_AVAILABLE = True
except Exception:
    TfidfVectorizer = None
    AgglomerativeClustering = None
    cosine_similarity = None
    _SKLEARN_AVAILABLE = False

# Local helpers
from .normalize import normalize_text


# ---------------------------------------------------------------------
# Config & interfaces
# ---------------------------------------------------------------------
@dataclass
class EmbedderConfig:
    """Configuration for embedding backends."""
    backend: str = "tfidf"          # "openai" | "sbert" | "tfidf"
    model: str = "all-MiniLM-L6-v2" # for sbert; ignored for tfidf
    batch_size: int = 64
    normalize: bool = True          # L2-normalize vectors (recommended)


class BaseEmbedder:
    """Abstract embedder interface."""
    name: str = "base"

    def __init__(self, config: EmbedderConfig | None = None):
        self.config = config or EmbedderConfig()

    def embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError("embed() must be implemented by subclasses.")

    @staticmethod
    def _l2_normalize(mat: np.ndarray) -> np.ndarray:
        if mat.size == 0:
            return mat
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


# ---------------------------------------------------------------------
# Concrete embedders
# ---------------------------------------------------------------------
class TfidfEmbedder(BaseEmbedder):
    """Deterministic TF-IDF embedding (fallback when neural embeddings are unavailable)."""

    def __init__(self, config: EmbedderConfig | None = None, ngram_range=(1, 2), min_df: int = 1):
        super().__init__(config)
        self.name = "tfidf"
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for TF-IDF embeddings.")
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df)

    def embed(self, texts: List[str]) -> np.ndarray:
        texts_norm = [normalize_text(t) for t in texts]
        X = self.vectorizer.fit_transform(texts_norm)
        mat = X.toarray().astype(np.float32)
        return self._l2_normalize(mat) if self.config.normalize else mat


class SBertEmbedder(BaseEmbedder):
    """Sentence-Transformers embedder (local, no network)."""

    def __init__(self, config: EmbedderConfig | None = None):
        super().__init__(config)
        self.name = "sbert"
        if not _SBERT_AVAILABLE:
            raise ImportError("sentence-transformers is not installed.")
        model_name = self.config.model or "all-MiniLM-L6-v2"
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        texts_norm = [normalize_text(t) for t in texts]
        # encode returns np.ndarray (float32)
        mat = self.model.encode(texts_norm, batch_size=self.config.batch_size, show_progress_bar=False, normalize_embeddings=False)
        mat = mat.astype(np.float32, copy=False)
        return self._l2_normalize(mat) if self.config.normalize else mat


class OpenAIEmbedder(BaseEmbedder):
    """
    Thin wrapper over an OpenAI embeddings client implemented in src/integration/openai_client.py.
    Expected client interface:
        client = OpenAIClient(api_key=..., model="text-embedding-3-small")
        vectors = client.embed_texts(texts: List[str]) -> List[List[float]]
    """

    def __init__(self, client, config: EmbedderConfig | None = None):
        super().__init__(config)
        self.name = "openai"
        self.client = client

    def embed(self, texts: List[str]) -> np.ndarray:
        texts_norm = [normalize_text(t) for t in texts]
        if not hasattr(self.client, "embed_texts"):
            raise AttributeError("OpenAI client must implement embed_texts(texts: List[str]) -> List[List[float]]")
        vecs = self.client.embed_texts(texts_norm, model=getattr(self.config, "model", None))
        mat = np.asarray(vecs, dtype=np.float32)
        return self._l2_normalize(mat) if self.config.normalize else mat


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------
def get_embedder(config: EmbedderConfig, *, openai_client=None) -> BaseEmbedder:
    """
    Return an embedder instance based on the given config.
    """
    backend = (config.backend or "tfidf").lower()
    if backend == "openai":
        if openai_client is None:
            raise ValueError("openai_client is required when backend='openai'.")
        return OpenAIEmbedder(openai_client, config)
    if backend == "sbert":
        return SBertEmbedder(config)
    # default
    return TfidfEmbedder(config)


# ---------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------
@dataclass
class ClusterConfig:
    algorithm: str = "hdbscan"   # "hdbscan" | "agglomerative"
    min_cluster_size: int = 5
    min_samples: int = 5
    max_clusters_hint: Optional[int] = None  # used by agglomerative
    random_state: int = 42                   # for deterministic agglomerative behavior where applicable


def cluster_embeddings(
    emb: np.ndarray,
    cfg: ClusterConfig,
) -> np.ndarray:
    """
    Cluster embedding matrix into labels.
    Returns labels array of shape (n,), with -1 indicating noise (for HDBSCAN).
    """
    n = emb.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    algo = (cfg.algorithm or "hdbscan").lower()
    if algo == "hdbscan":
        if not _HDBSCAN_AVAILABLE:
            warnings.warn("HDBSCAN not installed; falling back to AgglomerativeClustering.")
        else:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=max(2, int(cfg.min_cluster_size)),
                min_samples=max(1, int(cfg.min_samples)),
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=False,
            )
            labels = clusterer.fit_predict(emb)
            return labels.astype(int)

    # Fallback: Agglomerative with heuristic cluster count ~ sqrt(n)
    if not _SKLEARN_AVAILABLE or AgglomerativeClustering is None:
        # single cluster
        return np.zeros(n, dtype=int)

    k = cfg.max_clusters_hint or max(2, int(math.sqrt(n)))
    try:
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(emb)
        return labels.astype(int)
    except Exception:
        return np.zeros(n, dtype=int)


# ---------------------------------------------------------------------
# High-level pipeline for keywords
# ---------------------------------------------------------------------
def cluster_keywords_with_embeddings(
    df: pd.DataFrame,
    text_col: str = "kw_norm",
    ms_col: str = "avg_monthly_searches",
    embedder: Optional[BaseEmbedder] = None,
    embedder_config: Optional[EmbedderConfig] = None,
    cluster_config: Optional[ClusterConfig] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Compute embeddings for keywords and cluster them.
    Returns (df_out, embeddings), where df_out adds ['cluster_id','cluster_head'].
    """
    if df is None or df.empty or text_col not in df.columns:
        return pd.DataFrame(columns=list(df.columns) + ["cluster_id", "cluster_head"]), np.zeros((0, 1), dtype=np.float32)

    texts = df[text_col].astype(str).tolist()
    # Build embedder if needed
    if embedder is None:
        embedder = get_embedder(embedder_config or EmbedderConfig())
    emb = embedder.embed(texts)

    # Cluster
    ccfg = cluster_config or ClusterConfig()
    labels = cluster_embeddings(emb, ccfg)

    d = df.copy()
    d["cluster_id"] = labels

    # Pick cluster head by max monthly searches (or frequency)
    if ms_col in d.columns:
        d["_ms"] = pd.to_numeric(d[ms_col], errors="coerce").fillna(0)
    else:
        d["_ms"] = 0

    heads = {}
    for cid, sub in d.groupby("cluster_id"):
        # For HDBSCAN, cid may be -1 (noise): treat all as one group and pick best term
        sub_sorted = sub.sort_values("_ms", ascending=False)
        head = sub_sorted.iloc[0][text_col] if not sub_sorted.empty else f"Cluster {cid}"
        heads[int(cid)] = head

    d["cluster_head"] = d["cluster_id"].map(heads).astype(str)

    # Clean temp column
    d = d.drop(columns=["_ms"], errors="ignore")
    return d, emb


# ---------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------
def pairwise_cosine(a: np.ndarray, b: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute cosine similarity matrix. If `b` is None, returns square matrix a·a^T.
    """
    if cosine_similarity is None:
        # manual cosine
        x = a
        y = b if b is not None else a
        x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
        y_norm = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-9)
        return np.clip(x_norm @ y_norm.T, -1.0, 1.0)
    return cosine_similarity(a, b) if b is not None else cosine_similarity(a)


def assign_to_heads(
    queries: List[str],
    heads: List[str],
    *,
    embedder: Optional[BaseEmbedder] = None,
    embedder_config: Optional[EmbedderConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign each query to the most similar head using embeddings + cosine.
    Returns (best_index, best_similarity).
    """
    if not queries or not heads:
        return np.array([], dtype=int), np.array([], dtype=float)

    if embedder is None:
        embedder = get_embedder(embedder_config or EmbedderConfig())

    q_emb = embedder.embed(queries)
    h_emb = embedder.embed(heads)
    sims = pairwise_cosine(q_emb, h_emb)
    best_idx = np.argmax(sims, axis=1)
    best_sim = sims[np.arange(sims.shape[0]), best_idx]
    return best_idx.astype(int), best_sim.astype(float)
