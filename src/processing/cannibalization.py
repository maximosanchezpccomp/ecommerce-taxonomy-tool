# src/processing/cannibalization.py
# Node-level cannibalization detection (overlap/duplication) and redirect suggestions.
# IMPORTANT: Code in English.

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional deps (guarded)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    _SKLEARN_OK = True
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None
    _SKLEARN_OK = False

from .normalize import normalize_text, tokenize


# -------------------------------------------------------------------
# Public configuration
# -------------------------------------------------------------------
@dataclass
class CannibalizationParams:
    """
    Thresholds for detecting near-duplicate/overlapping nodes.
    """
    jaccard_thr: float = 0.60          # token Jaccard on names
    cosine_thr: float = 0.85           # TF-IDF cosine on names
    kw_overlap_thr: float = 0.50       # Jaccard overlap of keyword sets (if provided)
    allow_contained_slugs: bool = False  # if False, '/a/b' inside '/a/b/' counts as conflict
    prefer_score_over_ms: bool = True    # for winner selection


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------
def detect_cannibalization(
    nodes: List[dict],
    keywords_to_node: Optional[pd.DataFrame] = None,
    params: Optional[CannibalizationParams] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect overlapping/duplicate nodes and propose consolidations (redirects).

    Inputs
    ------
    nodes : List[dict]
        Single-market node list with fields: id, parent_id, name, slug, ms, score.
    keywords_to_node : DataFrame (optional)
        Mapping table with columns: ['keyword','node_id', 'similarity', 'intent'].
        Used to compute keyword set overlaps between nodes.
    params : CannibalizationParams

    Returns
    -------
    pairs_df : DataFrame
        Columns:
          ['node_a','name_a','slug_a','ms_a','score_a',
           'node_b','name_b','slug_b','ms_b','score_b',
           'sim_name_jaccard','sim_name_cosine','kw_overlap_jaccard',
           'slug_conflict','reason','recommendation','winner','loser']
    redirects_df : DataFrame
        Columns:
          ['from_slug','to_slug','reason','from_node','to_node']
    """
    p = params or CannibalizationParams()
    if not nodes or len(nodes) < 2:
        return pd.DataFrame(), pd.DataFrame()

    df_nodes = _nodes_df(nodes)
    # Only compare among siblings (same parent) to reduce false positives
    pairs = []
    for parent_id, group in df_nodes.groupby("parent_id", dropna=False):
        pairs.extend(list(itertools.combinations(group.index.tolist(), 2)))

    if not pairs:
        return pd.DataFrame(), pd.DataFrame()

    # Precompute TF-IDF cosine on names (optional)
    cos_matrix = _cosine_on_names(df_nodes["name"].tolist()) if _SKLEARN_OK else None

    # Build keyword sets per node (optional)
    kw_sets = _keywords_sets_by_node(keywords_to_node) if isinstance(keywords_to_node, pd.DataFrame) else {}

    rows = []
    for i, j in pairs:
        a = df_nodes.loc[i]; b = df_nodes.loc[j]

        # Name similarities
        jac = _jaccard_tokens(a["name"], b["name"])
        cos = float(cos_matrix[i, j]) if cos_matrix is not None else np.nan

        # Keyword Jaccard
        kwj = _set_jaccard(kw_sets.get(a["id"], set()), kw_sets.get(b["id"], set()))

        # Slug conflict?
        slug_conf = _slug_conflict(a["slug"], b["slug"], allow_contained=p.allow_contained_slugs)

        # Flag as potential cannibalization
        is_sim_name = (jac >= p.jaccard_thr) or (not np.isnan(cos) and cos >= p.cosine_thr)
        is_kw_overlap = kwj >= p.kw_overlap_thr if kw_sets else False

        if is_sim_name or is_kw_overlap or slug_conf:
            winner, loser = _pick_winner(a, b, prefer_score=p.prefer_score_over_ms)
            reason_bits = []
            if is_sim_name:
                reason_bits.append("similar names")
            if is_kw_overlap:
                reason_bits.append("keyword overlap")
            if slug_conf:
                reason_bits.append("slug conflict")
            reason = ", ".join(reason_bits) if reason_bits else "potential overlap"

            rec = "merge_redirect" if (is_sim_name and (is_kw_overlap or slug_conf)) else "disambiguate_labels"

            rows.append({
                "node_a": a["id"], "name_a": a["name"], "slug_a": a["slug"], "ms_a": a["ms"], "score_a": a["score"],
                "node_b": b["id"], "name_b": b["name"], "slug_b": b["slug"], "ms_b": b["ms"], "score_b": b["score"],
                "sim_name_jaccard": round(jac, 3),
                "sim_name_cosine": (round(cos, 3) if not np.isnan(cos) else np.nan),
                "kw_overlap_jaccard": round(kwj, 3) if kwj >= 0 else np.nan,
                "slug_conflict": bool(slug_conf),
                "reason": reason,
                "recommendation": rec,
                "winner": winner["id"] if winner is not None else None,
                "loser": loser["id"] if loser is not None else None,
            })

    pairs_df = pd.DataFrame(rows).sort_values(
        by=["recommendation", "sim_name_jaccard"],
        ascending=[True, False]
    ).reset_index(drop=True)

    redirects_df = _redirects_from_pairs(pairs_df, df_nodes)

    return pairs_df, redirects_df


# -------------------------------------------------------------------
# Internals
# -------------------------------------------------------------------
def _nodes_df(nodes: List[dict]) -> pd.DataFrame:
    d = pd.DataFrame([{
        "id": n.get("id"),
        "parent_id": n.get("parent_id"),
        "name": str(n.get("name", "") or ""),
        "slug": str(n.get("slug", "") or ""),
        "ms": int(n.get("ms", 0) or 0),
        "score": float(n.get("score", 0.0) or 0.0),
    } for n in nodes])
    # Normalize for safer comparisons
    d["_name_norm"] = d["name"].map(normalize_text)
    d["_slug_norm"] = d["slug"].str.strip().str.lower()
    return d


def _jaccard_tokens(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta and not tb:
        return 0.0
    return len(ta & tb) / float(len(ta | tb))


def _cosine_on_names(names: List[str]) -> Optional[np.ndarray]:
    if not _SKLEARN_OK or not names:
        return None
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform([normalize_text(x) for x in names])
    sims = cosine_similarity(X)
    return sims


def _keywords_sets_by_node(df: pd.DataFrame) -> Dict[str, set]:
    """
    Build {node_id -> set of normalized keywords} from keywords_to_node mapping.
    """
    out: Dict[str, set] = {}
    if df is None or df.empty:
        return out
    if not {"keyword", "node_id"}.issubset(set(df.columns)):
        return out
    tmp = df.copy()
    tmp["kw_norm"] = tmp["keyword"].astype(str).map(normalize_text)
    for nid, sub in tmp.groupby("node_id"):
        out[str(nid)] = set(sub["kw_norm"].tolist())
    return out


def _set_jaccard(a: set, b: set) -> float:
    if not a and not b:
        return -1.0  # "unknown" if sets missing; helps downstream display
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def _slug_conflict(a: str, b: str, *, allow_contained: bool) -> bool:
    if not a or not b:
        return False
    a = a.strip().lower().rstrip("/")
    b = b.strip().lower().rstrip("/")
    if a == b:
        return True
    if not allow_contained and (a.startswith(b + "/") or b.startswith(a + "/")):
        return True
    # Also treat last-segment collisions like '/x/foo' vs '/y/foo'
    seg_a = a.split("/")[-1]
    seg_b = b.split("/")[-1]
    return seg_a == seg_b


def _pick_winner(a: pd.Series, b: pd.Series, *, prefer_score: bool = True) -> Tuple[pd.Series, pd.Series]:
    """
    Choose canonical node to keep. Prioritize score, then ms, then shorter slug.
    """
    def key(r: pd.Series) -> Tuple[float, int, int]:
        return (float(r.get("score", 0.0)), int(r.get("ms", 0)), -len(str(r.get("slug", ""))))
    ka = key(a); kb = key(b)
    if prefer_score:
        if ka > kb:
            return a, b
        if kb > ka:
            return b, a
    # fallback: compare ms first
    if int(a["ms"]) > int(b["ms"]):
        return a, b
    if int(b["ms"]) > int(a["ms"]):
        return b, a
    # final tie-breaker: shorter slug (cleaner URL)
    return (a, b) if len(a["slug"]) <= len(b["slug"]) else (b, a)


def _redirects_from_pairs(pairs_df: pd.DataFrame, nodes_df: pd.DataFrame) -> pd.DataFrame:
    if pairs_df is None or pairs_df.empty:
        return pd.DataFrame(columns=["from_slug", "to_slug", "reason", "from_node", "to_node"])

    rows = []
    for _, r in pairs_df.iterrows():
        if r.get("recommendation") != "merge_redirect":
            continue
        loser_id = r.get("loser")
        winner_id = r.get("winner")
        if not loser_id or not winner_id:
            continue
        try:
            los = nodes_df[nodes_df["id"] == loser_id].iloc[0]
            win = nodes_df[nodes_df["id"] == winner_id].iloc[0]
        except Exception:
            continue
        reason = f"Consolidate duplicate/overlap: {r.get('reason','')}; keep '{win['name']}'"
        rows.append({
            "from_slug": str(los["slug"]),
            "to_slug": str(win["slug"]),
            "reason": reason,
            "from_node": str(loser_id),
            "to_node": str(winner_id),
        })
    return pd.DataFrame(rows)
