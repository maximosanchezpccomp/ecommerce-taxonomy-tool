# src/ingestion/competitor_loader.py
# Competitor structures loader: URLs/paths/titles CSV/Excel and XML sitemaps.
# IMPORTANT: Code in English.

from __future__ import annotations

import io
import re
import csv
import gzip
import json
import zipfile
import pathlib
import unicodedata
from typing import Iterable, List, Optional, Tuple, Union

import pandas as pd
from xml.etree import ElementTree as ET


FileLike = Union[io.BytesIO, io.StringIO, "UploadedFile", pathlib.Path, str]


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------
def read_competitor_paths(source: Union[FileLike, Iterable[FileLike]]) -> pd.DataFrame:
    """
    Load competitor navigation data from one or multiple sources:
      - CSV / Excel with columns (any subset): url, path, title
      - XML sitemap (urlset or sitemapindex)
      - GZ-compressed XML sitemap (.xml.gz)
      - Plain text file with one URL per line

    Returns a standardized DataFrame with columns:
        ['domain', 'url', 'path', 'title', 'resource_type', 'depth']

    Notes
    -----
    * This function does not fetch remote URLs. It only parses uploaded files
      or file-like objects provided by the user.
    * If a sitemap index is provided, only entries present inside the file are parsed.
      It will NOT download nested sitemaps from the web.
    """
    if source is None:
        return _empty_df()

    sources: List[FileLike] = list(source) if _is_iterable_but_not_str(source) else [source]
    frames: List[pd.DataFrame] = []

    for item in sources:
        try:
            kind = _detect_source_kind(item)
            if kind in ("csv", "excel"):
                frames.append(_read_csv_or_excel(item))
            elif kind in ("xml", "xml_gz"):
                frames.append(_read_sitemap_xml_any(item))
            elif kind == "text":
                frames.append(_read_plaintext_urls(item))
            else:
                # Last resort: try as CSV
                frames.append(_read_csv_or_excel(item))
        except Exception:
            # On parsing errors, skip this item gracefully
            continue

    if not frames:
        return _empty_df()

    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return _empty_df()

    df = _standardize_and_enrich(df)
    # Drop exact duplicates
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    return df


# -------------------------------------------------------------------
# Parsers
# -------------------------------------------------------------------
def _read_csv_or_excel(f: FileLike) -> pd.DataFrame:
    name = _filename_of(f).lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(f)
    else:
        # Try common CSV encodings/separators
        _seek_safely(f, 0)
        try:
            df = pd.read_csv(f, encoding="utf-8", sep=None, engine="python")
        except Exception:
            _seek_safely(f, 0)
            try:
                df = pd.read_csv(f, encoding="latin-1", sep=None, engine="python")
            except Exception:
                _seek_safely(f, 0)
                df = pd.read_csv(f)  # let pandas guess

    if df is None or df.empty:
        return _empty_df()

    # Normalize columns
    df.columns = [_norm(c) for c in df.columns]
    # Map possible headers to canonical ones
    url_col = _find_col(df.columns, ["url", "enlace", "link", "adresse", "address"])
    path_col = _find_col(df.columns, ["path", "ruta", "chemin"])
    title_col = _find_col(df.columns, ["title", "titulo", "titre", "titolo"])

    out = pd.DataFrame()
    if url_col and url_col in df.columns:
        out["url"] = df[url_col].astype(str)
    else:
        out["url"] = ""

    if path_col and path_col in df.columns:
        out["path"] = df[path_col].astype(str)
    else:
        out["path"] = out["url"].map(_extract_path)

    if title_col and title_col in df.columns:
        out["title"] = df[title_col].astype(str)
    else:
        out["title"] = ""

    return out


def _read_sitemap_xml_any(f: FileLike) -> pd.DataFrame:
    name = _filename_of(f).lower()
    raw = None

    try:
        if name.endswith(".gz"):
            _seek_safely(f, 0)
            raw = gzip.decompress(_read_all_bytes(f))
        else:
            raw = _read_all_bytes(f)
    except Exception:
        # Try reading as text then encode
        try:
            raw_txt = _read_all_text(f)
            raw = raw_txt.encode("utf-8", errors="ignore")
        except Exception:
            return _empty_df()

    if not raw:
        return _empty_df()

    try:
        root = ET.fromstring(raw)
    except Exception:
        # Sometimes there are stray BOMs or comments — attempt a cleanup
        try:
            cleaned = _strip_non_xml_prefix(raw.decode("utf-8", errors="ignore")).encode("utf-8")
            root = ET.fromstring(cleaned)
        except Exception:
            return _empty_df()

    tag = _localname(root.tag)

    urls: List[str] = []
    titles: List[str] = []

    if tag.endswith("sitemapindex"):
        # Extract nested sitemap <loc> entries within the same file (no fetching)
        for sm in root.findall(".//{*}sitemap"):
            loc = sm.findtext(".//{*}loc") or ""
            if loc:
                # We cannot fetch; store the sitemap URL itself as a row marker
                urls.append(loc.strip())
                titles.append("Sitemap index entry")
    elif tag.endswith("urlset"):
        for u in root.findall(".//{*}url"):
            loc = u.findtext(".//{*}loc") or ""
            if loc:
                urls.append(loc.strip())
                # Some sitemaps carry <news:title> or <image:title>
                t = u.findtext(".//{*}news/{*}title") or u.findtext(".//{*}image/{*}title") or ""
                titles.append(t.strip())
    else:
        return _empty_df()

    if not urls:
        return _empty_df()

    df = pd.DataFrame({"url": urls, "title": titles})
    df["path"] = df["url"].map(_extract_path)
    return df


def _read_plaintext_urls(f: FileLike) -> pd.DataFrame:
    text = _read_all_text(f)
    if not text:
        return _empty_df()

    # Accept JSON array of URLs or newline-separated
    text_stripped = text.strip()
    urls: List[str] = []
    if text_stripped.startswith("["):
        try:
            data = json.loads(text_stripped)
            if isinstance(data, list):
                urls = [str(x) for x in data]
        except Exception:
            urls = []
    if not urls:
        for line in text.splitlines():
            line = line.strip()
            if line and _looks_like_url(line):
                urls.append(line)

    if not urls:
        return _empty_df()

    df = pd.DataFrame({"url": urls})
    df["path"] = df["url"].map(_extract_path)
    df["title"] = ""
    return df


# -------------------------------------------------------------------
# Standardization & heuristics
# -------------------------------------------------------------------
def _standardize_and_enrich(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    # Ensure core columns
    for c in ["url", "path", "title"]:
        if c not in df.columns:
            df[c] = ""

    # Drop empty URL rows
    df = df[df["url"].astype(str).str.strip().ne("")]
    if df.empty:
        return _empty_df()

    # Derive domain, clean path, depth and resource type
    df["url"] = df["url"].astype(str).str.strip()
    df["path"] = df["path"].astype(str).apply(_clean_path_from_any)
    df["domain"] = df["url"].map(_extract_domain)
    df["depth"] = df["path"].apply(lambda p: max(0, len([x for x in p.strip("/").split("/") if x])))
    df["resource_type"] = df.apply(lambda r: _classify_type(r["path"], r.get("title", "")), axis=1)

    # Reorder columns
    cols = ["domain", "url", "path", "title", "resource_type", "depth"]
    rest = [c for c in df.columns if c not in cols]
    df = df[cols + rest]
    return df.reset_index(drop=True)


def _classify_type(path: str, title: str = "") -> str:
    """
    Very simple heuristic to classify URL path into:
    'category' | 'plp' | 'pdp' | 'content' | 'other'
    """
    p = path.lower()
    # Category hints
    if any(k in p for k in ["/category/", "/categoria/", "/categorias/", "/product-category/", "/c/"]):
        return "category"
    # PLP (listing) hints
    if any(k in p for k in ["/shop/", "/tienda/", "/products/", "/productos/", "/list", "/filter", "page=", "order="]):
        return "plp"
    # PDP (product detail) hints
    if any(k in p for k in ["/product/", "/producto/", "/prod/", "/p/"]):
        return "pdp"
    # Content / blog
    if any(k in p for k in ["/blog/", "/guides/", "/guia/", "/guía/", "/consejos/", "/news/", "/noticias/"]):
        return "content"
    # Fallback by pattern: long last segment with hyphens often is PDP
    last = p.strip("/").split("/")[-1] if p.strip("/") else ""
    if last and len(last) >= 16 and "-" in last:
        return "pdp"
    return "other"


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
def _is_iterable_but_not_str(x) -> bool:
    if isinstance(x, (str, bytes)):
        return False
    try:
        iter(x)  # type: ignore
        return True
    except Exception:
        return False


def _filename_of(f: FileLike) -> str:
    if hasattr(f, "name"):
        try:
            return str(getattr(f, "name"))
        except Exception:
            return ""
    if isinstance(f, (str, pathlib.Path)):
        return str(f)
    return ""


def _seek_safely(f, pos: int) -> None:
    try:
        f.seek(pos)
    except Exception:
        pass


def _read_all_bytes(f: FileLike) -> bytes:
    if isinstance(f, (str, pathlib.Path)):
        with open(f, "rb") as h:
            return h.read()
    if hasattr(f, "read"):
        _seek_safely(f, 0)
        data = f.read()
        # Streamlit UploadedFile.read() returns bytes
        if isinstance(data, bytes):
            return data
        if isinstance(data, str):
            return data.encode("utf-8", errors="ignore")
    return b""


def _read_all_text(f: FileLike) -> str:
    if isinstance(f, (str, pathlib.Path)):
        try:
            with open(f, "r", encoding="utf-8") as h:
                return h.read()
        except Exception:
            with open(f, "r", encoding="latin-1", errors="ignore") as h:
                return h.read()
    if hasattr(f, "read"):
        _seek_safely(f, 0)
        data = f.read()
        if isinstance(data, bytes):
            try:
                return data.decode("utf-8")
            except Exception:
                return data.decode("latin-1", errors="ignore")
        if isinstance(data, str):
            return data
    return ""


def _strip_non_xml_prefix(text: str) -> str:
    # Remove BOM or junk before the first '<'
    i = text.find("<")
    return text[i:] if i >= 0 else text


def _norm(s: str) -> str:
    s = str(s or "")
    s = s.strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = re.sub(r"\s+", " ", s)
    return s


def _find_col(cols: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols_norm = {_norm(c): c for c in cols}
    for cand in candidates:
        for norm, raw in cols_norm.items():
            if cand in norm:
                return raw
    return None


def _looks_like_url(s: str) -> bool:
    return bool(re.match(r"^https?://", s.strip(), flags=re.IGNORECASE))


def _extract_path(url: str) -> str:
    url = str(url or "").strip()
    if not url:
        return ""
    # Remove protocol and domain, keep path
    try:
        # remove scheme
        u = re.sub(r"^[a-z]+://", "", url, flags=re.IGNORECASE)
        # remove query and fragment
        u = u.split("?", 1)[0].split("#", 1)[0]
        # path after first slash
        path = "/" + u.split("/", 1)[1] if "/" in u else "/"
        return _clean_path_from_any(path)
    except Exception:
        return "/"


def _extract_domain(url: str) -> str:
    url = str(url or "").strip()
    try:
        u = re.sub(r"^[a-z]+://", "", url, flags=re.IGNORECASE)
        host = u.split("/", 1)[0]
        return host.lower()
    except Exception:
        return ""


def _clean_path_from_any(path: str) -> str:
    p = str(path or "").strip()
    if not p:
        return "/"
    # Remove protocol/domain remnants if present
    p = re.sub(r"^[a-z]+://[^/]+", "", p, flags=re.IGNORECASE)
    # Strip query/fragment
    p = p.split("?", 1)[0].split("#", 1)[0]
    if not p.startswith("/"):
        p = "/" + p
    # Collapse multiple slashes
    p = re.sub(r"/{2,}", "/", p)
    return p


def _localname(tag: str) -> str:
    """Return XML tag without namespace."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["domain", "url", "path", "title", "resource_type", "depth"])
