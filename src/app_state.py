# src/app_state.py
# Centralized session & configuration helpers for the eCommerce Taxonomy Tool.
# IMPORTANT: Code in English.

from __future__ import annotations

import io
import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Streamlit is optional at import time to keep CLI use-cases working.
try:
    import streamlit as st  # type: ignore
    _ST_AVAILABLE = True
except Exception:
    st = None  # type: ignore
    _ST_AVAILABLE = False

# Optional YAML support for external preset/config files
try:
    import yaml  # type: ignore
    _YAML_AVAILABLE = True
except Exception:
    _YAML_AVAILABLE = False


# -----------------------------
# Constants
# -----------------------------
SESSION_VERSION = "0.1.0"

SESSION_KEYS = {
    "params": "params",
    "gkp_data": "gkp_data",                   # Dict[str, DataFrame]
    "coverage_by_market": "coverage_by_market",
    "clusters_preview": "clusters_preview",   # Dict[str, DataFrame]
    "taxonomy_nodes": "taxonomy_nodes",       # Dict[str, List[dict]]
    "cannibal_pairs": "cannibal_pairs",       # Dict[str, DataFrame]
    "gsc_queries": "gsc_queries",             # DataFrame
    "gsc_pages": "gsc_pages",                 # DataFrame
    "competitors": "competitors",             # DataFrame
    "catalog": "catalog",                     # DataFrame
    "audit_log": "audit_log",                 # List[dict]
}

DEFAULT_MARKETS = ["es", "pt", "fr", "it"]


# -----------------------------
# Default parameters (safe baseline)
# -----------------------------
def default_params() -> Dict[str, Any]:
    """Return safe, opinionated default parameters."""
    return {
        "preset": "Calidad (por defecto)",
        "seo_horizon": "mixto",  # "informacional" | "mixto" | "transaccional"
        "markets": DEFAULT_MARKETS.copy(),
        "clustering": {
            "algorithm": "tfidf+agglomerative",  # "embeddings+hdbscan" | "tfidf+agglomerative" | "custom"
            "min_cluster_size": 5,
            "min_samples": 5,
        },
        "thresholds": {
            "max_levels": 3,
            "max_items_per_level": 12,
            "min_ms_quantile": {"es": 0.60, "default": 0.50},
            "cannibalization": {"jaccard": 0.60, "cosine": 0.85},
        },
        "naming": {
            "slug_case": "kebab",
            "strip_accents": True,
            "lowercase": True,
            "max_slug_len": 80,
            "dedupe_synonyms": True,
            "forbidden_terms": True,
        },
        "filters": {
            "mobile_first": True,
            "max_filters_core": 7,
            "min_core_filters_per_plp": 3,
            "auto_values_from_catalog": True,
        },
        "metadata": {
            "use_llm": True,
            "model": "gpt-5-thinking",
            "temperature": 0.0,
            "length_targets": {"title": [60, 65], "meta": [140, 160], "copy_words": [100, 150]},
            "add_faq": True,
        },
        "intent": {
            "use_llm_arbiter": True,
            "rules_weight": 0.5,
        },
        "serp_validation": {
            "enabled": False,
            "max_categories_to_check": 10,
        },
        "gsc": {
            "compare_current_vs_proposed": True,
            "date_range_days": 90,
            "country": "ES",
            "device": "all",
        },
    }


# -----------------------------
# Presets loader (optional YAML)
# -----------------------------
def _read_yaml(path: Path) -> Dict[str, Any]:
    if not _YAML_AVAILABLE:
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def load_presets_from_repo(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Attempt to load optional presets from configs/presets.yaml relative to repo root.
    """
    root = repo_root or Path.cwd()
    path = root / "configs" / "presets.yaml"
    return _read_yaml(path)


def apply_preset(base_params: Dict[str, Any], preset_name: str, presets_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Merge preset values (if available) into base params. Does not remove existing keys.
    """
    params = json.loads(json.dumps(base_params))  # deep copy
    presets_cfg = presets_cfg or {}
    preset = presets_cfg.get(preset_name.lower()) or presets_cfg.get(preset_name) or {}

    def _merge(dst: Dict[str, Any], src: Dict[str, Any]):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                _merge(dst[k], v)
            else:
                dst[k] = v

    if preset:
        _merge(params, preset)
    params["preset"] = preset_name
    return params


# -----------------------------
# Session accessors
# -----------------------------
def ensure_session_key(key: str, default_value):
    if not _ST_AVAILABLE:
        return default_value
    if key not in st.session_state:
        st.session_state[key] = default_value
    return st.session_state[key]


def get_params() -> Dict[str, Any]:
    """
    Return current params from session; initialize with defaults + optional preset file if empty.
    """
    if not _ST_AVAILABLE:
        return default_params()
    params = st.session_state.get(SESSION_KEYS["params"])
    if not params:
        base = default_params()
        # Try to apply preset from repo config (if any)
        cfg = load_presets_from_repo()
        preset_name = base.get("preset", "Calidad (por defecto)")
        merged = apply_preset(base, preset_name, cfg)
        st.session_state[SESSION_KEYS["params"]] = merged
        params = merged
    return params


def set_params(new_params: Dict[str, Any]) -> None:
    if not _ST_AVAILABLE:
        return
    st.session_state[SESSION_KEYS["params"]] = new_params
    add_audit("params_updated", {"preset": new_params.get("preset")} )


def get_market_data() -> Dict[str, pd.DataFrame]:
    return ensure_session_key(SESSION_KEYS["gkp_data"], {})


def set_market_data(per_market: Dict[str, pd.DataFrame]) -> None:
    if not _ST_AVAILABLE:
        return
    st.session_state[SESSION_KEYS["gkp_data"]] = per_market
    add_audit("gkp_loaded", {"markets": list(per_market.keys())})


def get_taxonomy_nodes() -> Dict[str, List[dict]]:
    return ensure_session_key(SESSION_KEYS["taxonomy_nodes"], {})


def set_taxonomy_nodes(nodes_by_market: Dict[str, List[dict]]) -> None:
    if not _ST_AVAILABLE:
        return
    st.session_state[SESSION_KEYS["taxonomy_nodes"]] = nodes_by_market
    add_audit("taxonomy_updated", {"markets": list(nodes_by_market.keys())})


def get_clusters_preview() -> Dict[str, pd.DataFrame]:
    return ensure_session_key(SESSION_KEYS["clusters_preview"], {})


def set_clusters_preview(clusters_by_market: Dict[str, pd.DataFrame]) -> None:
    if not _ST_AVAILABLE:
        return
    st.session_state[SESSION_KEYS["clusters_preview"]] = clusters_by_market


def get_gsc_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    q = ensure_session_key(SESSION_KEYS["gsc_queries"], pd.DataFrame())
    p = ensure_session_key(SESSION_KEYS["gsc_pages"], pd.DataFrame())
    return q, p


def set_gsc_data(queries_df: pd.DataFrame, pages_df: pd.DataFrame) -> None:
    if not _ST_AVAILABLE:
        return
    st.session_state[SESSION_KEYS["gsc_queries"]] = queries_df
    st.session_state[SESSION_KEYS["gsc_pages"]] = pages_df
    add_audit("gsc_loaded", {"queries": int(getattr(queries_df, "shape", [0])[0]), "pages": int(getattr(pages_df, "shape", [0])[0])})


def get_catalog() -> pd.DataFrame:
    return ensure_session_key(SESSION_KEYS["catalog"], pd.DataFrame())


def set_catalog(df: pd.DataFrame) -> None:
    if not _ST_AVAILABLE:
        return
    st.session_state[SESSION_KEYS["catalog"]] = df
    add_audit("catalog_loaded", {"rows": int(getattr(df, "shape", [0])[0])})


def get_competitors() -> pd.DataFrame:
    return ensure_session_key(SESSION_KEYS["competitors"], pd.DataFrame())


def set_competitors(df: pd.DataFrame) -> None:
    if not _ST_AVAILABLE:
        return
    st.session_state[SESSION_KEYS["competitors"]] = df
    add_audit("competitors_loaded", {"rows": int(getattr(df, "shape", [0])[0])})


def get_coverage_df() -> pd.DataFrame:
    return ensure_session_key(SESSION_KEYS["coverage_by_market"], pd.DataFrame())


def set_coverage_df(df: pd.DataFrame) -> None:
    if not _ST_AVAILABLE:
        return
    st.session_state[SESSION_KEYS["coverage_by_market"]] = df


def add_audit(event: str, data: Optional[Dict[str, Any]] = None) -> None:
    if not _ST_AVAILABLE:
        return
    log = st.session_state.get(SESSION_KEYS["audit_log"]) or []
    log.append({"t": int(time.time()), "event": event, "data": data or {}})
    st.session_state[SESSION_KEYS["audit_log"]] = log


def get_audit_log() -> List[Dict[str, Any]]:
    return ensure_session_key(SESSION_KEYS["audit_log"], [])


# -----------------------------
# Coverage & normalization utils
# -----------------------------
def normalize_text(s: str) -> str:
    import unicodedata
    s = str(s)
    s = s.lower()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return " ".join(s.split())


def ensure_kw_norm(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "kw_norm" not in d.columns:
        d["kw_norm"] = d["keyword"].astype(str).map(normalize_text)
    return d


def compute_coverage_by_market(per_market: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for mk, d in per_market.items():
        if d is None or d.empty:
            rows.append({"market": mk, "ms_total": 0, "keywords_count": 0})
            continue
        ms = int(pd.to_numeric(d.get("avg_monthly_searches", pd.Series(dtype=int)), errors="coerce").fillna(0).sum())
        rows.append({"market": mk, "ms_total": ms, "keywords_count": int(d.shape[0])})
    return pd.DataFrame(rows)


# -----------------------------
# Session snapshot (save/load)
# -----------------------------
def build_session_snapshot(markets: Optional[List[str]] = None) -> Dict[str, Any]:
    """Build a JSON-serializable snapshot of the current session."""
    params = get_params()
    nodes_all = get_taxonomy_nodes()
    coverage = get_coverage_df()

    if markets:
        nodes = {mk: nodes_all.get(mk, []) for mk in markets}
    else:
        nodes = nodes_all

    snap = {
        "version": SESSION_VERSION,
        "timestamp": int(time.time()),
        "params": params,
        "markets": markets or list(nodes.keys()),
        "taxonomy_nodes": nodes,
        "coverage_by_market": coverage.to_dict(orient="list") if isinstance(coverage, pd.DataFrame) else coverage,
        # Note: omit raw GKP/GSC data by default (can be large)
    }
    return snap


def session_snapshot_bytes(markets: Optional[List[str]] = None) -> bytes:
    snap = build_session_snapshot(markets)
    return json.dumps(snap, ensure_ascii=False, indent=2).encode("utf-8")


def restore_session_from_snapshot(data: Dict[str, Any]) -> None:
    """Restore a minimal set of keys from a previously saved snapshot."""
    if not _ST_AVAILABLE:
        return
    if not isinstance(data, dict):
        raise ValueError("Invalid session snapshot (not a dict).")

    if "params" in data:
        st.session_state[SESSION_KEYS["params"]] = data["params"]
    if "taxonomy_nodes" in data and isinstance(data["taxonomy_nodes"], dict):
        st.session_state[SESSION_KEYS["taxonomy_nodes"]] = data["taxonomy_nodes"]
    if "coverage_by_market" in data:
        try:
            # Try to reconstruct DataFrame
            cov = pd.DataFrame(data["coverage_by_market"])
        except Exception:
            cov = data["coverage_by_market"]
        st.session_state[SESSION_KEYS["coverage_by_market"]] = cov

    add_audit("session_restored", {"version": data.get("version")})


# -----------------------------
# Secrets & external creds
# -----------------------------
def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Read a secret from Streamlit secrets or environment variables."""
    if _ST_AVAILABLE and hasattr(st, "secrets"):
        try:
            val = st.secrets.get(key)  # type: ignore[attr-defined]
            if val is not None:
                return str(val)
        except Exception:
            pass
    return os.getenv(key, default)


def get_openai_config() -> Dict[str, Any]:
    return {
        "api_key": get_secret("OPENAI_API_KEY"),
        "model": get_params().get("metadata", {}).get("model", "gpt-5-thinking"),
        "temperature": float(get_params().get("metadata", {}).get("temperature", 0.0)),
    }


def get_gsc_config() -> Dict[str, Any]:
    return {
        "client_id": get_secret("GSC_CLIENT_ID"),
        "client_secret": get_secret("GSC_CLIENT_SECRET"),
        "redirect_uri": get_secret("GSC_REDIRECT_URI"),
        "property": get_secret("GSC_PROPERTY"),  # optional: site URL
        "scopes": ["https://www.googleapis.com/auth/webmasters.readonly"],
    }


# -----------------------------
# Clearing helpers
# -----------------------------
def clear(keys: Optional[List[str]] = None) -> None:
    """Clear selected session keys or all known keys."""
    if not _ST_AVAILABLE:
        return
    target_keys = keys or list(SESSION_KEYS.values())
    for k in target_keys:
        if k in st.session_state:
            del st.session_state[k]
    add_audit("session_cleared", {"keys": target_keys})
