# src/processing/normalize.py
# Text normalization, deduplication, and slug utilities (multilingual-friendly).
# IMPORTANT: Code in English.

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

# Optional lightweight stemmer (guarded)
try:
    import snowballstemmer  # type: ignore
    _STEM_AVAILABLE = True
except Exception:
    _STEM_AVAILABLE = False

# Optional string similarity (guarded)
try:
    from rapidfuzz import fuzz  # type: ignore
    _FUZZ_AVAILABLE = True
except Exception:
    _FUZZ_AVAILABLE = False


# ---------------------------------------------------
# Core text normalization
# ---------------------------------------------------
def normalize_text(s: str, *, strip_accents: bool = True, lower: bool = True) -> str:
    """
    Lowercase, trim, remove accents (optional), collapse whitespace, keep alphanumerics and spaces.
    Designed to be deterministic across ES/PT/FR/IT.
    """
    if s is None:
        return ""
    s = str(s).strip()
    if strip_accents:
        s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    if lower:
        s = s.lower()
    # Replace separators with space, collapse repeated whitespace
    s = re.sub(r"[^\w\s/+-]", " ", s)  # keep word chars, slash, plus, hyphen
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s: str) -> List[str]:
    s = normalize_text(s)
    # Split on space and slashes (useful for paths)
    toks = re.split(r"[\/\s]+", s)
    return [t for t in toks if t]


def ngrams(tokens: Sequence[str], n: int = 2) -> List[str]:
    if n <= 1:
        return list(tokens)
    return [" ".join(tokens[i : i + n]) for i in range(0, max(0, len(tokens) - n + 1))]


# ---------------------------------------------------
# Lemmatization/Stemming (optional, safe fallbacks)
# ---------------------------------------------------
def _get_stemmer(language: str = "es"):
    if not _STEM_AVAILABLE:
        return None
    # Map ISO-ish codes to Snowball names
    lang_map = {"es": "spanish", "pt": "portuguese", "fr": "french", "it": "italian", "en": "english"}
    snow_lang = lang_map.get(language.lower(), "english")
    try:
        return snowballstemmer.stemmer(snow_lang)
    except Exception:
        return None


def stem_text(s: str, language: str = "es") -> str:
    """
    Very light stemming; only used if snowballstemmer is available.
    If not, returns normalized text unchanged.
    """
    stemmer = _get_stemmer(language)
    if stemmer is None:
        return normalize_text(s)
    toks = tokenize(s)
    stems = stemmer.stemWords(toks)  # type: ignore[attr-defined]
    return " ".join(stems)


# ---------------------------------------------------
# Slug utilities
# ---------------------------------------------------
def slugify(
    s: str,
    *,
    case: str = "kebab",         # "kebab" (default) or "snake"
    strip_accents: bool = True,
    lowercase: bool = True,
    max_len: int = 80,
) -> str:
    """
    Convert a label to a URL-safe slug. Keeps '/' segments for category paths.
    """
    if s is None:
        return ""
    s_norm = normalize_text(s, strip_accents=strip_accents, lower=lowercase)
    # Replace separators according to case
    sep = "-" if case == "kebab" else "_"
    s_norm = re.sub(r"[^\w/]+", sep, s_norm)  # word chars or slash; others to sep
    s_norm = re.sub(fr"{re.escape(sep)}+", sep, s_norm)
    # Trim sep around slashes and ends
    s_norm = re.sub(fr"{re.escape(sep)}/", "/", s_norm)
    s_norm = re.sub(fr"/{re.escape(sep)}", "/", s_norm)
    s_norm = s_norm.strip(sep).strip("/")
    # Enforce max length per segment to avoid extremely long URLs
    parts = [p[:max_len] for p in s_norm.split("/")]
    out = "/".join(parts)
    return out


# ---------------------------------------------------
# Synonym/lexicon mapping (per market)
# ---------------------------------------------------
def apply_lexicon(term: str, lexicon: Dict[str, Dict[str, List[str]]], concept_key: Optional[str] = None) -> str:
    """
    Map synonyms to a preferred label using a simple lexicon structure:

    lexicon example (YAML/JSON):
    {
      "CONCEPT_VIDEO_DOORBELL": {
        "preferred": "Videoportero",
        "synonyms": ["timbre inteligente", "doorbell", "portero de vídeo"]
      },
      "CONCEPT_PEEPHOLE_DIGITAL": {
        "preferred": "Mirilla digital",
        "synonyms": ["judas digital", "judas electronico", "oeilleton numerique"]
      }
    }

    If `concept_key` is provided, only that concept is considered; otherwise all concepts are checked.
    Matching is case/diacritics-insensitive based on `normalize_text`.
    """
    if not lexicon:
        return term

    norm = normalize_text(term)
    concepts = [concept_key] if concept_key in (lexicon or {}) else list((lexicon or {}).keys())

    for key in concepts:
        entry = lexicon.get(key) or {}
        preferred = entry.get("preferred", "").strip()
        synonyms = [normalize_text(x) for x in (entry.get("synonyms") or []) if x]
        if not preferred:
            continue
        # Direct match to any synonym or preferred
        if norm == normalize_text(preferred) or norm in synonyms:
            return preferred
        # Token-level contains (avoid over-aggressive replacements)
        for syn in synonyms:
            if syn and f" {syn} " in f" {norm} ":
                return preferred
    return term


# ---------------------------------------------------
# Dedupe & fuzzy grouping
# ---------------------------------------------------
def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def fuzzy_ratio(a: str, b: str) -> float:
    """
    Return similarity in [0,1]. Uses rapidfuzz if available; else token Jaccard.
    """
    if _FUZZ_AVAILABLE:
        try:
            return float(fuzz.token_set_ratio(a, b)) / 100.0  # type: ignore
        except Exception:
            pass
    return jaccard(tokenize(a), tokenize(b))


@dataclass
class DedupeConfig:
    """
    Heuristics to deduplicate near-duplicate keywords while keeping the strongest representative.
    """
    similarity_threshold: float = 0.90   # 0.90 with rapidfuzz; consider 0.80 if only Jaccard
    prefer_higher_ms: bool = True        # keep row with higher monthly searches


def dedupe_keywords_df(df: pd.DataFrame, *, key_col: str = "keyword", ms_col: str = "avg_monthly_searches",
                       config: Optional[DedupeConfig] = None) -> pd.DataFrame:
    """
    Deduplicate keywords using fuzzy similarity. Keeps a single representative per near-duplicate group.

    Returns a DataFrame with an added 'kw_norm' column and duplicates removed.
    """
    if df is None or df.empty or key_col not in df.columns:
        return pd.DataFrame(columns=[key_col, ms_col, "kw_norm"])

    cfg = config or DedupeConfig()
    d = df.copy()
    d[key_col] = d[key_col].astype(str)
    if ms_col not in d.columns:
        d[ms_col] = 0
    d["kw_norm"] = d[key_col].map(normalize_text)

    # Sort by monthly searches desc to keep strongest first
    d = d.sort_values(ms_col, ascending=False).reset_index(drop=True)

    keep_mask = [True] * len(d)
    reps: List[int] = []

    for i in range(len(d)):
        if not keep_mask[i]:
            continue
        reps.append(i)
        a = d.loc[i, "kw_norm"]
        # Compare only to later rows (O(n^2) but OK for 1e3 scale; switch to LSH/HNSW for larger)
        for j in range(i + 1, len(d)):
            if not keep_mask[j]:
                continue
            b = d.loc[j, "kw_norm"]
            sim = fuzzy_ratio(a, b)
            if sim >= cfg.similarity_threshold:
                keep_mask[j] = False

    out = d[keep_mask].reset_index(drop=True)
    return out


# ---------------------------------------------------
# Convenience for pipelines
# ---------------------------------------------------
def ensure_kw_norm(df: pd.DataFrame, key_col: str = "keyword") -> pd.DataFrame:
    d = df.copy()
    if key_col not in d.columns:
        raise ValueError(f"Column '{key_col}' not found")
    if "kw_norm" not in d.columns:
        d["kw_norm"] = d[key_col].map(normalize_text)
    return d


def apply_market_lexicon_column(df: pd.DataFrame, lexicon: Dict[str, Dict[str, List[str]]], *,
                                src_col: str = "keyword", dst_col: str = "keyword_canonical") -> pd.DataFrame:
    """
    Apply lexicon mapping to a DataFrame column, creating a canonicalized label.
    """
    d = df.copy()
    d[dst_col] = d[src_col].map(lambda x: apply_lexicon(str(x), lexicon))
    return d
