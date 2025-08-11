# src/ingestion/gsc_loader.py
# Google Search Console (GSC) ZIP loader (robust to locales and formats).
# IMPORTANT: Code in English.

from __future__ import annotations

import io
import re
import zipfile
from typing import Iterable, Optional, Tuple

import pandas as pd


# ----------------------------
# Public API
# ----------------------------
def read_gsc_zip(file_like) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read a standard Google Search Console export ZIP and return two DataFrames:
      - queries_df with columns: ['query','clicks','impressions','ctr','position']
      - pages_df   with columns: ['page','clicks','impressions','ctr','position']

    Notes
    -----
    * Works with CSV (comma) and TSV (tab) inside the ZIP.
    * Tolerant to UTF-8 and Latin-1 encodings.
    * Robust to localized headers (ES/PT/FR/IT/EN). Columns are standardized to English.
    * If multiple files match (e.g., multiple property folders), picks the largest by rows.

    Parameters
    ----------
    file_like : file-like object
        A Streamlit UploadedFile or any handle accepted by zipfile.ZipFile.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (queries_df, pages_df). Either may be empty if not found.
    """
    if file_like is None:
        return _empty_queries(), _empty_pages()

    try:
        zf = zipfile.ZipFile(file_like)
    except Exception:
        # Not a valid ZIP
        return _empty_queries(), _empty_pages()

    # Collect candidate CSV/TSV members
    members = [n for n in zf.namelist() if n.lower().endswith((".csv", ".tsv"))]

    # Heuristic: “queries” vs “pages” files by path/name
    query_cands = [m for m in members if _looks_like_queries_filename(m)]
    page_cands = [m for m in members if _looks_like_pages_filename(m)]

    # Fallback: if we didn't find by filename, consider all CSV/TSV and disambiguate by header
    if not query_cands or not page_cands:
        for m in members:
            try:
                df_tmp = _read_member_autodetect(zf, m)
            except Exception:
                continue
            kind = _detect_kind_by_headers(df_tmp.columns)
            if kind == "queries" and m not in query_cands:
                query_cands.append(m)
            elif kind == "pages" and m not in page_cands:
                page_cands.append(m)

    # Pick largest by rows if multiple
    q_member = _pick_largest_member(zf, query_cands)
    p_member = _pick_largest_member(zf, page_cands)

    queries_df = _standardize_queries(_read_member_autodetect(zf, q_member)) if q_member else _empty_queries()
    pages_df = _standardize_pages(_read_member_autodetect(zf, p_member)) if p_member else _empty_pages()

    return queries_df, pages_df


# ----------------------------
# Internal helpers
# ----------------------------
def _empty_queries() -> pd.DataFrame:
    return pd.DataFrame(columns=["query", "clicks", "impressions", "ctr", "position"])


def _empty_pages() -> pd.DataFrame:
    return pd.DataFrame(columns=["page", "clicks", "impressions", "ctr", "position"])


def _looks_like_queries_filename(name: str) -> bool:
    n = name.lower()
    # Common patterns across locales
    return any(k in n for k in ["queries", "consultas", "consulta", "recherches", "ricerche", "pesquisas"])


def _looks_like_pages_filename(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in ["pages", "páginas", "paginas", "pagina", "pagine", "urls", "url"])


def _read_member_autodetect(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    """
    Read a CSV/TSV member with automatic delimiter and encoding detection.
    """
    with zf.open(member) as f:
        raw = f.read()

    # Try UTF-8 with pandas delimiter sniffing
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            buf = io.StringIO(raw.decode(enc, errors="strict"))
            # sep=None enables python engine sniffing
            df = pd.read_csv(buf, sep=None, engine="python")
            if not df.empty:
                return df
        except Exception:
            continue

    # Last resort: explicit comma then tab
    for enc in ("utf-8", "latin-1"):
        for sep in (",", "\t"):
            try:
                buf = io.StringIO(raw.decode(enc, errors="ignore"))
                df = pd.read_csv(buf, sep=sep)
                if not df.empty:
                    return df
            except Exception:
                continue

    return pd.DataFrame()


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def _detect_kind_by_headers(cols: Iterable[str]) -> Optional[str]:
    ncols = [_norm(c) for c in cols]

    # Signals for queries
    has_query = any(x in " ".join(ncols) for x in ["query", "consulta", "recherche", "ricerca", "pesquisa"])
    # Signals for pages
    has_page = any(x in " ".join(ncols) for x in ["page", "página", "pagina", "url"])

    # Prefer specific matches
    if has_query and not has_page:
        return "queries"
    if has_page and not has_query:
        return "pages"

    # Ambiguous → None
    return None


def _find_col(cols: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols_norm = {_norm(c): c for c in cols}
    for cand in candidates:
        for norm, raw in cols_norm.items():
            if cand in norm:
                return raw
    return None


def _to_float(series: pd.Series) -> pd.Series:
    """
    Convert localized numeric strings to float.
    Handles '1 234', '1.234', '1,234', '12,34', and percentages.
    """
    s = series.astype(str).str.strip()
    # Remove percent sign but remember it to convert to 0-1 if needed
    is_pct = s.str.contains("%", regex=False)
    s = s.str.replace("%", "", regex=False)

    # Remove currency/letters
    s = s.str.replace(r"[^\d,.\-]", "", regex=True)

    def _coerce(v: str) -> float:
        if v.count(",") and v.count("."):
            # Use last separator as decimal
            if v.rfind(",") > v.rfind("."):
                v = v.replace(".", "")
                v = v.replace(",", ".")
            else:
                v = v.replace(",", "")
        elif "," in v and "." not in v:
            # Single comma → decimal
            v = v.replace(".", "")
            v = v.replace(",", ".")
        else:
            # If many dots, clean thousands
            parts = v.split(".")
            if len(parts) > 2:
                v = "".join(parts[:-1]) + "." + parts[-1]
        try:
            return float(v)
        except Exception:
            return float("nan")

    out = s.apply(_coerce)
    # If original had %, convert to 0-1
    out[is_pct.fillna(False)] = out[is_pct.fillna(False)] / 100.0
    return out


def _standardize_queries(df_in: pd.DataFrame) -> pd.DataFrame:
    if df_in is None or df_in.empty:
        return _empty_queries()
    df = df_in.copy()
    # Identify columns across locales
    q_col = _find_col(df.columns, ["query", "consulta", "recherche", "ricerca", "pesquisa"])
    clicks_col = _find_col(df.columns, ["clicks", "clics", "cliques", "klicks"])
    imps_col = _find_col(df.columns, ["impressions", "impresiones", "impressions (fr)", "impressioni", "impressoes", "impressions"])
    ctr_col = _find_col(df.columns, ["ctr"])
    pos_col = _find_col(df.columns, ["position", "posición", "posicion", "posizione", "posição"])

    out = pd.DataFrame()
    out["query"] = df[q_col].astype(str) if q_col else ""
    out["clicks"] = _to_float(df[clicks_col]).fillna(0.0).astype(float) if clicks_col else 0.0
    out["impressions"] = _to_float(df[imps_col]).fillna(0.0).astype(float) if imps_col else 0.0
    out["ctr"] = _to_float(df[ctr_col]).fillna(0.0).astype(float) if ctr_col else 0.0
    out["position"] = _to_float(df[pos_col]).fillna(0.0).astype(float) if pos_col else 0.0

    # Cast clicks/impressions to int when safe
    try:
        out["clicks"] = out["clicks"].round(0).astype(int)
        out["impressions"] = out["impressions"].round(0).astype(int)
    except Exception:
        pass

    # Drop empty queries
    out = out[out["query"].astype(str).str.strip().ne("")]
    out = out.reset_index(drop=True)
    return out


def _standardize_pages(df_in: pd.DataFrame) -> pd.DataFrame:
    if df_in is None or df_in.empty:
        return _empty_pages()
    df = df_in.copy()
    # Identify columns across locales
    p_col = _find_col(df.columns, ["page", "página", "pagina", "url", "adress", "adresse"])
    clicks_col = _find_col(df.columns, ["clicks", "clics", "cliques", "klicks"])
    imps_col = _find_col(df.columns, ["impressions", "impresiones", "impressioni", "impressoes", "impressions"])
    ctr_col = _find_col(df.columns, ["ctr"])
    pos_col = _find_col(df.columns, ["position", "posición", "posicion", "posizione", "posição"])

    out = pd.DataFrame()
    out["page"] = df[p_col].astype(str) if p_col else ""
    out["clicks"] = _to_float(df[clicks_col]).fillna(0.0).astype(float) if clicks_col else 0.0
    out["impressions"] = _to_float(df[imps_col]).fillna(0.0).astype(float) if imps_col else 0.0
    out["ctr"] = _to_float(df[ctr_col]).fillna(0.0).astype(float) if ctr_col else 0.0
    out["position"] = _to_float(df[pos_col]).fillna(0.0).astype(float) if pos_col else 0.0

    # Cast clicks/impressions to int when safe
    try:
        out["clicks"] = out["clicks"].round(0).astype(int)
        out["impressions"] = out["impressions"].round(0).astype(int)
    except Exception:
        pass

    # Drop empty pages
    out = out[out["page"].astype(str).str.strip().ne("")]
    out = out.reset_index(drop=True)
    return out


def _pick_largest_member(zf: zipfile.ZipFile, members: list[str]) -> Optional[str]:
    """Pick the member that yields the most rows after parsing."""
    if not members:
        return None
    best_name, best_rows = None, -1
    for m in members:
        try:
            df = _read_member_autodetect(zf, m)
            rows = int(getattr(df, "shape", [0])[0])
            if rows > best_rows:
                best_rows = rows
                best_name = m
        except Exception:
            continue
    return best_name
