# src/ingestion/gkp_loader.py
# Robust Google Keyword Planner (GKP) loader for CSV/Excel in multiple locales.
# IMPORTANT: Code in English.

from __future__ import annotations

import io
import re
import unicodedata
from typing import Dict, Iterable, Optional

import pandas as pd


# ----------------------------
# Public API
# ----------------------------
def read_gkp_any(file_like) -> pd.DataFrame:
    """
    Read a Google Keyword Planner export (CSV or Excel) in various locales and return a
    standardized DataFrame with (at least) these columns:

      - keyword (str)
      - avg_monthly_searches (int)
      - competition (float in [0,1] or categorical if not numeric)
      - top_of_page_bid_low (float)
      - top_of_page_bid_high (float)
      - language (str)
      - location (str)

    Missing columns are created with sensible defaults (0 / empty strings).
    The function is tolerant to UTF-16/UTF-8 encodings and tab/comma separators.

    Parameters
    ----------
    file_like : file-like
        Streamlit UploadedFile, file handle, or pathlib.Path-like accepted by pandas.

    Returns
    -------
    pd.DataFrame
        Standardized GKP dataframe.
    """
    if file_like is None:
        return _empty_result()

    # Try Excel then CSV variations
    df = None
    name = getattr(file_like, "name", "")
    try:
        if name.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_like)
        else:
            # First try UTF-16 + tab (common in GKP)
            _seek_safely(file_like, 0)
            try:
                df = pd.read_csv(file_like, encoding="utf-16", sep="\t")
            except Exception:
                # Fallback to UTF-8 + comma
                _seek_safely(file_like, 0)
                df = pd.read_csv(file_like, encoding="utf-8", sep=",")
    except Exception:
        # Last resort generic parser
        _seek_safely(file_like, 0)
        df = pd.read_csv(file_like, engine="python")

    if df is None or df.empty:
        return _empty_result()

    df = _standardize_gkp_dataframe(df)
    return df


def read_gkp_path(path: str) -> pd.DataFrame:
    """
    Convenience wrapper to read from a filesystem path (for CLI/pytest).
    """
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        # Try UTF-16 + tab, then UTF-8 + comma
        try:
            df = pd.read_csv(path, encoding="utf-16", sep="\t")
        except Exception:
            df = pd.read_csv(path, encoding="utf-8", sep=",")
    return _standardize_gkp_dataframe(df)


# ----------------------------
# Internal helpers
# ----------------------------
def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "keyword",
            "avg_monthly_searches",
            "competition",
            "top_of_page_bid_low",
            "top_of_page_bid_high",
            "language",
            "location",
        ]
    )


def _seek_safely(f, pos: int) -> None:
    try:
        f.seek(pos)
    except Exception:
        pass


def _norm(s: str) -> str:
    s = str(s or "")
    s = s.strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = re.sub(r"\s+", " ", s)
    return s


def _is_numlike_series(x: pd.Series) -> bool:
    try:
        return pd.to_numeric(x, errors="coerce").notna().mean() > 0.7
    except Exception:
        return False


def _clean_numeric_str_to_float(series: pd.Series) -> pd.Series:
    """
    Handle strings like '1.200', '1,200', '1 200', '12,34', '€1,20' and convert to float.
    """
    s = (
        series.astype(str)
        .str.replace(r"[^\d,.\-]", "", regex=True)  # keep digits, signs, separators
        .str.replace(r"\s+", "", regex=True)
    )

    # If both comma and dot appear, assume comma is thousands and dot is decimal (or vice versa).
    # Heuristic: if last separator is comma, treat comma as decimal; if dot, treat dot as decimal.
    def _to_float(v: str) -> float:
        if v.count(",") and v.count("."):
            # Decide by last occurrence
            if v.rfind(",") > v.rfind("."):
                v = v.replace(".", "")
                v = v.replace(",", ".")
            else:
                v = v.replace(",", "")
        else:
            # Single separator: if comma present, use as decimal
            if "," in v and "." not in v:
                v = v.replace(".", "")
                v = v.replace(",", ".")
            else:
                # if only dots present, remove thousands dots except last decimal
                parts = v.split(".")
                if len(parts) > 2:
                    v = "".join(parts[:-1]) + "." + parts[-1]
        try:
            return float(v)
        except Exception:
            return float("nan")

    return s.apply(_to_float)


def _column_mapping(cols: Iterable[str]) -> Dict[str, str]:
    """
    Build a mapping from raw columns to standardized names using multilingual heuristics.
    """
    norm_cols = {_norm(c): c for c in cols}

    def find_first(candidates: Iterable[str]) -> Optional[str]:
        for cand in candidates:
            for norm, raw in norm_cols.items():
                if cand in norm:
                    return raw
        return None

    # Candidate substrings per field (ES/PT/FR/IT/EN)
    kw_candidates = [
        "keyword", "palabra clave", "palabras clave", "consulta", "termo", "termos", "terme", "termes", "parola", "parole",
    ]
    ms_candidates = [
        "avg monthly searches", "promedio de busquedas mensuales", "busquedas mensuales",
        "moyenne de recherches mensuelles", "nombre moyen de recherches", "moy de recherches",
        "media de ricerche mensili", "medie ricerche mensili",
        "media de pesquisas mensais", "pesquisas mensais",
        "monthly searches",
    ]
    comp_candidates = [
        "competition", "competencia", "concorrenza", "concurrence", "concorrencia"
    ]
    bid_low_candidates = [
        "top of page bid (low range)", "puja de parte superior de la pagina (rango bajo)",
        "enchere haut de page (fourchette basse)", "offerta in prima pagina (intervallo basso)",
        "lance na parte superior da pagina (intervalo baixo)", "low bid"
    ]
    bid_high_candidates = [
        "top of page bid (high range)", "puja de parte superior de la pagina (rango alto)",
        "enchere haut de page (fourchette haute)", "offerta in prima pagina (intervallo alto)",
        "lance na parte superior da pagina (intervalo alto)", "high bid"
    ]
    lang_candidates = [
        "language", "idioma", "lingua", "langue", "idioma de destino", "idiomas"
    ]
    loc_candidates = [
        "location", "ubicacion", "pais", "localizacion", "localidade", "localizacao",
        "lieu", "pays", "location targeting", "target location"
    ]

    mapping: Dict[str, str] = {}

    mapping["keyword"] = find_first(kw_candidates) or next(iter(cols))
    mapping["avg_monthly_searches"] = find_first(ms_candidates) or ""
    mapping["competition"] = find_first(comp_candidates) or ""
    mapping["top_of_page_bid_low"] = find_first(bid_low_candidates) or ""
    mapping["top_of_page_bid_high"] = find_first(bid_high_candidates) or ""
    mapping["language"] = find_first(lang_candidates) or ""
    mapping["location"] = find_first(loc_candidates) or ""

    return mapping


def _standardize_gkp_dataframe(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    # Normalize headers
    df.columns = [str(c).strip() for c in df.columns]
    mapping = _column_mapping(df.columns)

    # Ensure keyword column exists
    kw_col = mapping.get("keyword")
    if not kw_col or kw_col not in df.columns:
        # fallback to first column
        kw_col = df.columns[0]

    out = pd.DataFrame()
    out["keyword"] = df[kw_col].astype(str).str.strip()

    # Avg monthly searches
    ms_col = mapping.get("avg_monthly_searches")
    if ms_col and ms_col in df.columns:
        ms_series = _clean_numeric_str_to_float(df[ms_col]).fillna(0)
        out["avg_monthly_searches"] = ms_series.round(0).astype(int)
    else:
        out["avg_monthly_searches"] = 0

    # Competition can be categorical ("High/Medium/Low") or numeric in some exports
    comp_col = mapping.get("competition")
    if comp_col and comp_col in df.columns:
        comp_raw = df[comp_col]
        if _is_numlike_series(comp_raw):
            out["competition"] = pd.to_numeric(comp_raw, errors="coerce").fillna(0.0).astype(float)
        else:
            # Map text to ordinal: Low=0.33, Medium=0.66, High=1.0
            comp_map = {
                "low": 0.33, "baja": 0.33, "faible": 0.33, "basso": 0.33, "baixa": 0.33,
                "medium": 0.66, "media": 0.66, "moyenne": 0.66, "medio": 0.66,
                "high": 1.0, "alta": 1.0, "elevee": 1.0, "elevée": 1.0, "alto": 1.0, "alta (pt)": 1.0
            }
            out["competition"] = comp_raw.astype(str).map(lambda x: comp_map.get(_norm(x), None))
            # If mapping failed widely, keep original string column
            if out["competition"].isna().mean() > 0.7:
                out["competition"] = comp_raw.astype(str)
            else:
                out["competition"] = out["competition"].fillna(0.0).astype(float)
    else:
        out["competition"] = 0.0

    # Top of page bids (low/high)
    low_col = mapping.get("top_of_page_bid_low")
    high_col = mapping.get("top_of_page_bid_high")
    out["top_of_page_bid_low"] = _clean_numeric_str_to_float(df[low_col]).fillna(0.0).astype(float) if (low_col and low_col in df.columns) else 0.0
    out["top_of_page_bid_high"] = _clean_numeric_str_to_float(df[high_col]).fillna(0.0).astype(float) if (high_col and high_col in df.columns) else 0.0

    # Language & Location
    lang_col = mapping.get("language")
    loc_col = mapping.get("location")
    out["language"] = df[lang_col].astype(str) if (lang_col and lang_col in df.columns) else ""
    out["location"] = df[loc_col].astype(str) if (loc_col and loc_col in df.columns) else ""

    # Drop empty/NaN keywords
    out = out[out["keyword"].astype(str).str.strip().ne("")]
    out = out.reset_index(drop=True)
    return out
