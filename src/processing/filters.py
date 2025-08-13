# src/processing/filters.py
# Filter (faceted navigation) designer from catalog + keyword signals.
# IMPORTANT: Code in English.

from __future__ import annotations

import re
import math
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .normalize import normalize_text


# -------------------------------------------------------------------
# Data structures
# -------------------------------------------------------------------
@dataclass
class FilterDef:
    name: str
    type: str  # "in" | "range" | "boolean"
    values: List[str]
    order: int = 0
    visibility: str = "core"  # "core" | "advanced" | "hidden_mobile"


# -------------------------------------------------------------------
# Localization (labels per market)
# -------------------------------------------------------------------
_LBL = {
    "brand": {"es": "Marca", "pt": "Marca", "fr": "Marque", "it": "Marca"},
    "price": {"es": "Precio", "pt": "Preço", "fr": "Prix", "it": "Prezzo"},
    "resolution": {"es": "Resolución", "pt": "Resolução", "fr": "Résolution", "it": "Risoluzione"},
    "connectivity": {"es": "Conectividad", "pt": "Conectividade", "fr": "Connectivité", "it": "Connettività"},
    "power": {"es": "Alimentación", "pt": "Alimentação", "fr": "Alimentation", "it": "Alimentazione"},
    "storage": {"es": "Almacenamiento", "pt": "Armazenamento", "fr": "Stockage", "it": "Archiviazione"},
    "screen": {"es": "Tamaño de pantalla", "pt": "Tamanho do ecrã", "fr": "Taille d’écran", "it": "Dimensione schermo"},
    "frequency": {"es": "Frecuencia (Hz)", "pt": "Frequência (Hz)", "fr": "Fréquence (Hz)", "it": "Frequenza (Hz)"},
    "capacity": {"es": "Capacidad", "pt": "Capacidade", "fr": "Capacité", "it": "Capacità"},
    "panel": {"es": "Tipo de panel", "pt": "Tipo de painel", "fr": "Type de dalle", "it": "Tipo di pannello"},
    "ai_motion": {"es": "Detección de movimiento / IA", "pt": "Deteção de movimento / IA", "fr": "Détection de mouvement / IA", "it": "Rilevamento movimento / IA"},
    "angle": {"es": "Ángulo de visión", "pt": "Ângulo de visão", "fr": "Angle de vue", "it": "Angolo di visione"},
    "assistant": {"es": "Compatibilidad asistentes", "pt": "Compatibilidade assistentes", "fr": "Compatibilité assistants", "it": "Compatibilità assistenti"},
    "color": {"es": "Color/Acabado", "pt": "Cor/Acabamento", "fr": "Couleur/Finition", "it": "Colore/Finitura"},
    "type": {"es": "Tipo", "pt": "Tipo", "fr": "Type", "it": "Tipo"},
}


def _L(key: str, market: str) -> str:
    return _LBL.get(key, {}).get((market or "es").lower(), _LBL.get(key, {}).get("es", key))


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------
def design_filters(
    nodes: List[dict],
    *,
    market: str = "es",
    catalog_df: Optional[pd.DataFrame] = None,
    keywords_df: Optional[pd.DataFrame] = None,
    max_values_per_filter: int = 12,
    min_distinct_for_in: int = 2,
) -> Dict[str, List[dict]]:
    """
    Propose a set of filters (facets) for each subcategory node.

    Inputs
    ------
    nodes : List[dict]
        Taxonomy nodes for a single market (root + children). We generate filters only for children.
    market : str
        'es'|'pt'|'fr'|'it' for localized labels.
    catalog_df : pd.DataFrame (optional)
        Product snapshot with columns like:
            - brand|marca
            - price|precio
            - name|nombre|title
            - attribute columns (free-form)
    keywords_df : pd.DataFrame (optional)
        Keyword list with 'keyword' (and optionally the subset for each node if you pre-filter).
        Used to mine long-tail attribute hints.

    Returns
    -------
    Dict[str, List[dict]]
        Mapping node_id -> list of filter dicts with schema:
            {"name": str, "type": "in|range|boolean", "values": [str], "order": int, "visibility": "core|advanced|hidden_mobile"}
    """
    if not nodes:
        return {}

    mk = (market or "es").lower()
    children = [n for n in nodes if n.get("parent_id")]

    # Pre-compute global signals from catalog/keywords to reuse per node
    cat_signals = _mine_catalog_signals(catalog_df, mk, max_values_per_filter=max_values_per_filter)
    kw_signals = _mine_keyword_signals(keywords_df, mk, max_values=max_values_per_filter)

    result: Dict[str, List[dict]] = {}

    for i, node in enumerate(children):
        node_name = str(node.get("name", ""))
        # 1) Always include Brand + Price (core)
        filters: List[FilterDef] = []
        filters.append(FilterDef(name=_L("brand", mk), type="in", values=cat_signals.get("brand_values", []), order=1, visibility="core"))
        filters.append(FilterDef(name=_L("price", mk), type="range", values=[], order=2, visibility="core"))

        # 2) Node-specific heuristics (based on node label + available signals)
        # For the "mirillas/videoporteros/timbres" vertical we bias towards connectivity/resolution/power.
        node_norm = normalize_text(node_name)
        is_doorbell_vertical = any(k in node_norm for k in ["mirilla", "judas", "doorbell", "videoportero", "portero", "intercom", "timbre", "intercomunicador"])

        # Catalog-driven facets
        if cat_signals.get("resolution"):
            filters.append(FilterDef(name=_L("resolution", mk), type="in", values=cat_signals["resolution"], order=3, visibility="core"))
        if cat_signals.get("connectivity"):
            filters.append(FilterDef(name=_L("connectivity", mk), type="in", values=cat_signals["connectivity"], order=4, visibility="core" if is_doorbell_vertical else "advanced"))
        if cat_signals.get("power"):
            filters.append(FilterDef(name=_L("power", mk), type="in", values=cat_signals["power"], order=5, visibility="advanced" if is_doorbell_vertical else "advanced"))
        if cat_signals.get("angle"):
            filters.append(FilterDef(name=_L("angle", mk), type="in", values=cat_signals["angle"], order=6, visibility="advanced"))
        if cat_signals.get("assistant"):
            filters.append(FilterDef(name=_L("assistant", mk), type="in", values=cat_signals["assistant"], order=7, visibility="advanced"))
        if cat_signals.get("color"):
            filters.append(FilterDef(name=_L("color", mk), type="in", values=cat_signals["color"], order=8, visibility="hidden_mobile"))

        # 3) Keyword-driven (fallback/augmentation)
        # Merge with catalog-derived ensuring uniqueness and max size
        _merge_in_filter(filters, name=_L("resolution", mk), values=kw_signals.get("resolution", []), order_hint=3, visibility="core")
        _merge_in_filter(filters, name=_L("connectivity", mk), values=kw_signals.get("connectivity", []), order_hint=4, visibility="core" if is_doorbell_vertical else "advanced")
        _merge_in_filter(filters, name=_L("power", mk), values=kw_signals.get("power", []), order_hint=5, visibility="advanced")
        _merge_in_filter(filters, name=_L("assistant", mk), values=kw_signals.get("assistant", []), order_hint=7, visibility="advanced")
        _merge_in_filter(filters, name=_L("screen", mk), values=kw_signals.get("screen_inches", []), order_hint=9, visibility="advanced")

        # 4) AI/Motion detection boolean (useful in smart security)
        filters.append(FilterDef(name=_L("ai_motion", mk), type="boolean", values=[], order=10, visibility="advanced"))

        # Clean: drop empty "in" filters (values < min_distinct_for_in)
        filters = _prune_weak(filters, min_distinct_for_in)

        # Sort by 'order' then by name
        filters.sort(key=lambda f: (int(f.order), f.name.lower()))

        # Cap number of core filters (mobile-first)
        filters = _enforce_mobile_first(filters, max_core=7)

        result[node["id"]] = [asdict(f) for f in filters]

    return result


# -------------------------------------------------------------------
# Mining from catalog
# -------------------------------------------------------------------
def _mine_catalog_signals(
    catalog_df: Optional[pd.DataFrame],
    market: str,
    max_values_per_filter: int = 12,
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    if catalog_df is None or catalog_df.empty:
        return out

    cols = {c.lower(): c for c in catalog_df.columns}

    # Brand
    brand_col = None
    for cand in ["brand", "marca", "marque", "marca (it)"]:
        if cand in cols:
            brand_col = cols[cand]; break
    if brand_col:
        out["brand_values"] = _top_values(catalog_df[brand_col], k=max_values_per_filter)

    # Price
    for cand in ["price", "precio", "preco", "prix", "prezzo"]:
        if cand in cols:
            # range handled at UI; no discrete values needed
            break

    # Resolution (720p, 1080p, 2K, 4K)
    out["resolution"] = _mine_resolution(catalog_df, cols, k=max_values_per_filter)

    # Connectivity (WiFi/Ethernet/PoE/BT)
    out["connectivity"] = _mine_connectivity(catalog_df, cols, k=max_values_per_filter)

    # Power (Batería/PoE/230V/USB)
    out["power"] = _mine_power(catalog_df, cols, k=max_values_per_filter)

    # Angle of view (common discrete buckets)
    out["angle"] = _mine_angle(catalog_df, cols, k=max_values_per_filter)

    # Assistants (Alexa/Google/HomeKit)
    out["assistant"] = _mine_assistants(catalog_df, cols, k=max_values_per_filter)

    # Color / finish
    out["color"] = _mine_color(catalog_df, cols, k=max_values_per_filter)

    return out


def _top_values(series: pd.Series, k: int = 12) -> List[str]:
    vals = (
        series.dropna()
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    top = vals.value_counts().head(k).index.tolist()
    return [v for v in top if v and v.lower() not in ["nan", "none", ""]]


def _mine_resolution(df: pd.DataFrame, cols: Dict[str, str], k: int = 12) -> List[str]:
    text = _concat_text_cols(df, cols)
    patt = r"(720p|900p|1080p|2k|1440p|4k|ultra hd|full hd|superhd|2\.?5k|5k)"
    return _freq_from_regex(text, patt, k, norm_map={
        "ultra hd": "UHD",
        "full hd": "Full HD",
        "superhd": "SuperHD",
        "2k": "2K",
        "4k": "4K",
    })


def _mine_connectivity(df: pd.DataFrame, cols: Dict[str, str], k: int = 12) -> List[str]:
    text = _concat_text_cols(df, cols)
    # include Wi-Fi spellings and PoE
    patt = r"(wi[\-\s]?fi|wifi|ethernet|poe|bluetooth|2\.4ghz|5ghz|lan)"
    return _freq_from_regex(text, patt, k, norm_map={
        "wi fi": "Wi-Fi",
        "wifi": "Wi-Fi",
        "ethernet": "Ethernet",
        "poe": "PoE",
        "bluetooth": "Bluetooth",
        "2.4ghz": "2.4GHz",
        "5ghz": "5GHz",
        "lan": "LAN",
    })


def _mine_power(df: pd.DataFrame, cols: Dict[str, str], k: int = 12) -> List[str]:
    text = _concat_text_cols(df, cols)
    patt = r"(bater[ií]a|battery|pilas?|poe|usb|230v|220v|corriente|ac\b|dc\b|12v|24v|alimentaci[oó]n)"
    return _freq_from_regex(text, patt, k, norm_map={
        "bateria": "Batería",
        "batería": "Batería",
        "pilas": "Pilas",
        "poe": "PoE",
        "usb": "USB",
        "230v": "230V",
        "220v": "220V",
        "corriente": "Corriente",
        "ac": "AC",
        "dc": "DC",
        "12v": "12V",
        "24v": "24V",
        "alimentacion": "Alimentación",
        "alimentación": "Alimentación",
    })


def _mine_angle(df: pd.DataFrame, cols: Dict[str, str], k: int = 12) -> List[str]:
    text = _concat_text_cols(df, cols)
    patt = r"(\b1[0-9]{2}°|\b[6-9][0-9]°|\b120°|\b130°|\b140°|\b150°|\b160°|\b170°|\b180°)"
    return _freq_from_regex(text, patt, k)


def _mine_assistants(df: pd.DataFrame, cols: Dict[str, str], k: int = 12) -> List[str]:
    text = _concat_text_cols(df, cols)
    patt = r"(alexa|google (home|assistant)?|homekit|siri|tuya|ifttt|matter)"
    return _freq_from_regex(text, patt, k, norm_map={
        "google": "Google Assistant",
        "google home": "Google Home",
        "homekit": "Apple HomeKit",
        "ifttt": "IFTTT",
        "matter": "Matter",
    })


def _mine_color(df: pd.DataFrame, cols: Dict[str, str], k: int = 12) -> List[str]:
    text = _concat_text_cols(df, cols)
    # simple color list in ES + international
    patt = r"(negro|blanco|gris|plata|dorado|cromo|lat[oó]n|niquel|níquel|black|white|silver|gold|chrome)"
    return _freq_from_regex(text, patt, k, norm_map={
        "negro": "Negro",
        "blanco": "Blanco",
        "gris": "Gris",
        "plata": "Plata",
        "dorado": "Dorado",
        "cromo": "Cromo",
        "laton": "Latón",
        "latón": "Latón",
        "niquel": "Níquel",
        "níquel": "Níquel",
        "black": "Negro",
        "white": "Blanco",
        "silver": "Plata",
        "gold": "Dorado",
        "chrome": "Cromo",
    })


def _concat_text_cols(df: pd.DataFrame, cols_map: Dict[str, str]) -> pd.Series:
    cand_cols = [cols_map[c] for c in cols_map if c in ["name", "nombre", "title", "titulo", "description", "descripcion", "descrição", "descrizione"]]
    if not cand_cols:
        cand_cols = list(df.columns)  # last resort
    text = df[cand_cols].astype(str).agg(" ".join, axis=1)
    return text.str.lower().str.replace(r"\s+", " ", regex=True)


def _freq_from_regex(series: pd.Series, pattern: str, k: int, norm_map: Optional[Dict[str, str]] = None) -> List[str]:
    norm_map = norm_map or {}
    counter: Counter = Counter()
    for s in series.dropna().astype(str):
        for m in re.findall(pattern, s, flags=re.IGNORECASE):
            m = " ".join(m) if isinstance(m, tuple) else m
            key = _normalize_token(m)
            key = norm_map.get(key, key)
            counter[key] += 1
    vals = [v for v, _ in counter.most_common(k)]
    return vals


def _normalize_token(s: str) -> str:
    s = str(s or "").strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = re.sub(r"\s+", " ", s)
    return s


# -------------------------------------------------------------------
# Mining from keywords (long-tail attributes)
# -------------------------------------------------------------------
def _mine_keyword_signals(keywords_df: Optional[pd.DataFrame], market: str, max_values: int = 12) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    if keywords_df is None or keywords_df.empty or "keyword" not in keywords_df.columns:
        return out

    kws = keywords_df["keyword"].astype(str).str.lower()

    # Resolution
    out["resolution"] = _freq_from_regex(kws, r"(720p|900p|1080p|2k|1440p|4k|full hd|uhd|superhd)", max_values, norm_map={
        "full hd": "Full HD", "uhd": "UHD", "2k": "2K", "4k": "4K"
    })

    # Connectivity
    out["connectivity"] = _freq_from_regex(kws, r"(wi[\-\s]?fi|wifi|ethernet|poe|bluetooth|2\.4ghz|5ghz|lan)", max_values, norm_map={
        "wi fi": "Wi-Fi", "wifi": "Wi-Fi", "poe": "PoE", "ethernet": "Ethernet", "bluetooth": "Bluetooth", "2.4ghz": "2.4GHz", "5ghz": "5GHz", "lan": "LAN"
    })

    # Power
    out["power"] = _freq_from_regex(kws, r"(bater[ií]a|battery|pilas?|poe|usb|230v|220v|ac\b|dc\b|12v|24v)", max_values, norm_map={
        "bateria": "Batería", "batería": "Batería", "poe": "PoE", "usb": "USB", "230v": "230V", "220v": "220V", "ac": "AC", "dc": "DC", "12v":"12V","24v":"24V"
    })

    # Assistants
    out["assistant"] = _freq_from_regex(kws, r"(alexa|google (home|assistant)?|homekit|siri|tuya|ifttt|matter)", max_values, norm_map={
        "google": "Google Assistant", "google home": "Google Home", "homekit": "Apple HomeKit", "ifttt": "IFTTT", "matter": "Matter"
    })

    # Screen inches (extract values like 3.2", 7", 10.1")
    sizes = []
    for s in kws.dropna().astype(str):
        for m in re.findall(r"(\d{1,2}(\.\d)?)[\"”]?(\s*(inch|pulg|\"))", s):
            val = m[0]
            sizes.append(f'{val}"')
    if sizes:
        cnt = Counter(sizes)
        out["screen_inches"] = [v for v, _ in cnt.most_common(max_values)]

    return out


# -------------------------------------------------------------------
# Utilities for merging/pruning
# -------------------------------------------------------------------
def _find_filter(filters: List[FilterDef], name: str) -> Optional[FilterDef]:
    for f in filters:
        if f.name.strip().lower() == name.strip().lower():
            return f
    return None


def _merge_in_filter(filters: List[FilterDef], *, name: str, values: List[str], order_hint: int, visibility: str):
    if not values:
        return
    f = _find_filter(filters, name)
    if f is None:
        f = FilterDef(name=name, type="in", values=[], order=order_hint, visibility=visibility)
        filters.append(f)
    # Extend & dedupe preserving order by frequency (assumed in input)
    existing = set(v.lower() for v in f.values)
    for v in values:
        if v and v.lower() not in existing:
            f.values.append(v)
            existing.add(v.lower())
    # Cap to 12 values
    if len(f.values) > 12:
        f.values = f.values[:12]


def _prune_weak(filters: List[FilterDef], min_distinct_for_in: int) -> List[FilterDef]:
    cleaned: List[FilterDef] = []
    for f in filters:
        if f.type == "in":
            vals = [v for v in (f.values or []) if v]
            if len(vals) < min_distinct_for_in:
                continue
            f.values = vals
        cleaned.append(f)
    return cleaned


def _enforce_mobile_first(filters: List[FilterDef], max_core: int = 7) -> List[FilterDef]:
    core = [f for f in filters if f.visibility == "core"]
    advanced = [f for f in filters if f.visibility == "advanced"]
    hidden = [f for f in filters if f.visibility == "hidden_mobile"]

    if len(core) > max_core:
        # demote overflow to advanced keeping relative order
        demote = core[max_core:]
        for f in demote:
            f.visibility = "advanced"
        core = core[:max_core]

    # Reassemble preserving relative order by 'order'
    core.sort(key=lambda f: int(f.order))
    advanced.sort(key=lambda f: int(f.order))
    hidden.sort(key=lambda f: int(f.order))
    return core + advanced + hidden
