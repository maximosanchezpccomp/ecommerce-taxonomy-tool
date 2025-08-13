# src/processing/hierarchy.py
# Propose a market-aware category → subcategory hierarchy from clustered keywords.
# IMPORTANT: Code in English.

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .normalize import normalize_text, slugify, apply_lexicon


# ------------------------------------------------------------
# Config & constants
# ------------------------------------------------------------
LOCALIZED_ROOT_FALLBACK = {
    "es": "Tecnología",
    "pt": "Tecnologia",
    "fr": "Technologie",
    "it": "Tecnologia",
}

# Simple titlecase that keeps common all-caps acronyms untouched
def _smart_title(s: str) -> str:
    if not s:
        return s
    words = str(s).strip().split()
    out = []
    for w in words:
        if len(w) <= 3 and w.isupper():
            out.append(w)
        else:
            out.append(w.capitalize())
    return " ".join(out)


@dataclass
class HierarchyParams:
    """
    Parameters to control the hierarchy proposal.
    """
    max_levels: int = 3                    # currently we produce root + level-2 children
    max_items_per_level: int = 12          # UX guardrail
    seo_horizon: str = "mixto"             # "informacional" | "mixto" | "transaccional"
    slug_case: str = "kebab"               # "kebab" | "snake"
    strip_accents: bool = True
    lowercase: bool = True
    max_slug_len: int = 80
    # Optional lexicon per market to canonicalize names
    # Format:
    #   {"CONCEPT_KEY": {"preferred": "Mirilla digital", "synonyms": ["judas numerique", ...]}, ...}
    market_lexicons: Optional[Dict[str, Dict[str, Dict[str, List[str]]]]] = None
    # Optional forbidden terms list by market
    forbidden_terms: Optional[Dict[str, List[str]]] = None


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------
def propose_hierarchy(
    df_labeled: pd.DataFrame,
    selected_clusters: pd.DataFrame,
    *,
    market: str = "es",
    params: Optional[HierarchyParams] = None,
    root_name: Optional[str] = None,
    base_path: Optional[str] = None,
    competitors_df: Optional[pd.DataFrame] = None,
    catalog_df: Optional[pd.DataFrame] = None,
) -> List[Dict]:
    """
    Build a 2-level hierarchy (root + subcategories) for a given market.

    Inputs
    ------
    df_labeled:
        Keyword dataframe with at least: ['keyword','avg_monthly_searches','cluster_id','cluster_head'].
    selected_clusters:
        Aggregated clusters with ['cluster_id','cluster_head','ms_cluster'] filtered by quantile.
    market:
        'es'|'pt'|'fr'|'it'
    params:
        HierarchyParams for naming/SEO rules.
    root_name:
        Optional display name for the root (defaults to LOCALIZED_ROOT_FALLBACK[market]).
    base_path:
        Optional base path segment to prepend to slugs, e.g. '/seguridad-hogar' → '/seguridad-hogar/<slug-child>'.
        If omitted, slugs start at '/<root>/<child>'.
    competitors_df:
        Optional competitor table with at least ['path','title'] to inform score boosts & recommended examples.
    catalog_df:
        Optional catalog snapshot to compute coverage signals (brands/models count) (not required).

    Returns
    -------
    List[dict] of nodes matching the tool schema:
        {
          "id": str, "parent_id": str|None, "name": str, "slug": str, "intent": str,
          "ms": int, "score": float,
          "recommended_PLPs": [str],
          "filters": [],
          "seo": {"title": "", "h1": "", "meta": "", "faq": []}
        }
    """
    p = params or HierarchyParams()
    mk = (market or "es").lower()

    # Guards
    if df_labeled is None or df_labeled.empty or selected_clusters is None or selected_clusters.empty:
        return _minimal_root_tree(root_name or LOCALIZED_ROOT_FALLBACK.get(mk, "Tecnología"), market=mk, params=p)

    # Root node
    root_display = root_name or LOCALIZED_ROOT_FALLBACK.get(mk, "Tecnología")
    root_slug_base = (
        base_path.strip().rstrip("/") if base_path else f"/{slugify(root_display, case=p.slug_case, strip_accents=p.strip_accents, lowercase=p.lowercase, max_len=p.max_slug_len)}"
    )
    if not root_slug_base.startswith("/"):
        root_slug_base = "/" + root_slug_base

    root_id = _gen_id()
    nodes: List[Dict] = [
        {
            "id": root_id,
            "parent_id": None,
            "name": root_display,
            "slug": root_slug_base,
            "intent": "mixed",
            "ms": _safe_int(df_labeled.get("avg_monthly_searches")),
            "score": 1.0,
            "recommended_PLPs": [],
            "filters": [],
            "seo": {"title": "", "h1": root_display, "meta": "", "faq": []},
        }
    ]

    # Prepare helpers
    total_ms_selected = max(1, int(selected_clusters["ms_cluster"].sum()))
    lex = (p.market_lexicons or {}).get(mk) if p.market_lexicons else None
    forb = set((p.forbidden_terms or {}).get(mk, []))

    # Competitor assists
    comp_paths = competitors_df["path"].astype(str).tolist() if isinstance(competitors_df, pd.DataFrame) and "path" in competitors_df.columns else []
    comp_titles = competitors_df["title"].astype(str).tolist() if isinstance(competitors_df, pd.DataFrame) and "title" in competitors_df.columns else []
    comp_corpus = [normalize_text(x) for x in (comp_paths + comp_titles)]

    # Intent signal (if present on df_labeled) aggregated per cluster
    intent_map = _intent_by_cluster(df_labeled)

    # Build children from selected clusters
    selected = selected_clusters.sort_values("ms_cluster", ascending=False).head(int(p.max_items_per_level)).reset_index(drop=True)

    for _, row in selected.iterrows():
        cid = int(row["cluster_id"])
        head = str(row["cluster_head"])
        ms = int(row["ms_cluster"])

        # Canonicalize display name via lexicon if available
        display = _smart_title(apply_lexicon(head, lexicon=lex) if lex else head)
        # Block forbidden terms softly
        display = _strip_forbidden(display, forb)

        # Pick intent for the cluster (default "transactional" for PLPs)
        intent = intent_map.get(cid, "transactional")

        # Score: combine normalized search share + horizon weighting + competitor presence
        score = _score_child(
            ms=ms,
            total_ms_selected=total_ms_selected,
            intent=intent,
            seo_horizon=p.seo_horizon,
            child_label=display,
            comp_corpus=comp_corpus,
        )

        child_slug = f"{root_slug_base}/{slugify(display, case=p.slug_case, strip_accents=p.strip_accents, lowercase=p.lowercase, max_len=p.max_slug_len)}"
        rec_plps = _recommend_plps_for_child(display, child_slug, competitors_df)

        nodes.append({
            "id": _gen_id(),
            "parent_id": root_id,
            "name": display,
            "slug": child_slug,
            "intent": intent,
            "ms": int(ms),
            "score": round(float(score), 3),
            "recommended_PLPs": rec_plps,
            "filters": [],                 # filled later in filters module / UI
            "seo": {"title": "", "h1": display, "meta": "", "faq": []},
        })

    return nodes


# ------------------------------------------------------------
# Internals
# ------------------------------------------------------------
def _gen_id() -> str:
    import uuid
    return str(uuid.uuid4())


def _safe_int(series_or_val) -> int:
    try:
        if isinstance(series_or_val, (pd.Series, pd.DataFrame)):
            return int(pd.to_numeric(series_or_val, errors="coerce").fillna(0).sum())
        return int(series_or_val)
    except Exception:
        return 0


def _minimal_root_tree(name: str, market: str, params: HierarchyParams) -> List[Dict]:
    slug_root = f"/{slugify(name, case=params.slug_case, strip_accents=params.strip_accents, lowercase=params.lowercase, max_len=params.max_slug_len)}"
    return [{
        "id": _gen_id(),
        "parent_id": None,
        "name": name,
        "slug": slug_root,
        "intent": "mixed",
        "ms": 0,
        "score": 1.0,
        "recommended_PLPs": [],
        "filters": [],
        "seo": {"title": "", "h1": name, "meta": "", "faq": []},
    }]


def _intent_by_cluster(df_labeled: pd.DataFrame) -> Dict[int, str]:
    """
    If df_labeled contains 'intent' or 'intent_score', compute majority/mean per cluster.
    """
    out: Dict[int, str] = {}
    if "cluster_id" not in df_labeled.columns:
        return out

    # Prefer explicit 'intent' if present
    if "intent" in df_labeled.columns:
        for cid, sub in df_labeled.groupby("cluster_id"):
            # majority label
            lab = sub["intent"].astype(str).str.lower().value_counts().idxmax()
            out[int(cid)] = lab
        return out

    # Else map score to label
    if "intent_score" in df_labeled.columns:
        for cid, sub in df_labeled.groupby("cluster_id"):
            sc = float(pd.to_numeric(sub["intent_score"], errors="coerce").fillna(0.5).mean())
            out[int(cid)] = "transactional" if sc >= 2/3 else ("informational" if sc <= 1/3 else "mixed")
    return out


def _strip_forbidden(label: str, forbidden: set[str]) -> str:
    if not forbidden:
        return label
    t = normalize_text(label)
    for bad in forbidden:
        b = normalize_text(bad)
        if not b:
            continue
        # remove whole-word matches
        t = t.replace(f" {b} ", " ").strip()
    # Re-title for display
    return _smart_title(t)


def _score_child(
    *,
    ms: int,
    total_ms_selected: int,
    intent: str,
    seo_horizon: str,
    child_label: str,
    comp_corpus: List[str],
) -> float:
    """
    Composite score: normalized MS × intent weight × competitor presence bonus.
    """
    base = (ms / max(1, total_ms_selected))
    # Intent weight depending on horizon
    intent = (intent or "mixed").lower()
    horizon = (seo_horizon or "mixto").lower()

    intent_w = {
        "informacional": {"informational": 1.1, "mixed": 1.0, "transactional": 0.9},
        "mixto":         {"informational": 1.0, "mixed": 1.05, "transactional": 1.05},
        "transaccional": {"informational": 0.9, "mixed": 1.05, "transactional": 1.1},
    }.get(horizon, {"informational": 1.0, "mixed": 1.0, "transactional": 1.0}).get(intent, 1.0)

    # Competitor presence bonus (light): +2.5% per match up to +7.5%
    bonus = 1.0 + min(0.075, 0.025 * _competitor_presence(child_label, comp_corpus))

    return float(base) * float(intent_w) * float(bonus)


def _competitor_presence(label: str, comp_corpus: List[str]) -> int:
    """
    Count how many competitor entries (path/title) roughly contain the child label tokens.
    """
    if not comp_corpus:
        return 0
    t = normalize_text(label)
    toks = [tok for tok in t.split() if len(tok) >= 3]
    if not toks:
        return 0
    count = 0
    for doc in comp_corpus:
        if all(tok in doc for tok in toks):
            count += 1
            if count >= 3:
                break
    return count


def _recommend_plps_for_child(child_name: str, child_slug: str, competitors_df: Optional[pd.DataFrame]) -> List[str]:
    """
    Suggest up to 3 PLP-like paths based on competitor structures that look similar to this child.
    Fallback to the child_slug itself.
    """
    recs: List[str] = []
    if isinstance(competitors_df, pd.DataFrame) and not competitors_df.empty and "path" in competitors_df.columns:
        t = normalize_text(child_name)
        toks = [tok for tok in t.split() if len(tok) >= 3]
        if toks:
            cand = competitors_df.copy()
            cand["_score"] = cand["path"].astype(str).map(normalize_text).map(lambda p: _token_overlap_score(p, toks))
            cand = cand.sort_values("_score", ascending=False)
            cand = cand[cand["_score"] >= 0.5].head(3)
            recs = cand["path"].astype(str).tolist()
    if not recs:
        recs = [child_slug]
    return recs


def _token_overlap_score(doc: str, toks: List[str]) -> float:
    if not toks:
        return 0.0
    d = str(doc or "")
    score = sum(1 for t in toks if t in d) / float(len(toks))
    return float(score)
