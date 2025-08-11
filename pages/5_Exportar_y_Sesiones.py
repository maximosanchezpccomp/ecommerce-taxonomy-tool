# pages/5_Exportar_y_Sesiones.py
# Export & Sessions page (Streamlit multipage)
# IMPORTANT: Code in English. UI labels/messages in Spanish.

from __future__ import annotations

import io
import json
import time
import zipfile
import unicodedata
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Optional libs (guarded imports)
try:
    from pptx import Presentation
    PPTX_AVAILABLE = True
except Exception:
    PPTX_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None


# ---------------------------------------
# Helpers
# ---------------------------------------
def normalize_text(s: str) -> str:
    s = str(s)
    s = s.lower()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return " ".join(s.split())


def flatten_taxonomy(nodes: List[dict], market: str) -> pd.DataFrame:
    rows = []
    for n in nodes:
        rows.append({
            "market": market,
            "id": n.get("id", ""),
            "parent_id": n.get("parent_id") or "",
            "name": n.get("name", ""),
            "slug": n.get("slug", ""),
            "intent": n.get("intent", ""),
            "ms": int(n.get("ms", 0) or 0),
            "score": float(n.get("score", 0.0) or 0.0)
        })
    return pd.DataFrame(rows)


def flatten_filters(nodes: List[dict], market: str) -> pd.DataFrame:
    rows = []
    for n in nodes:
        for f in (n.get("filters") or []):
            # handle dict or dataclass-like
            name = f.get("name") if isinstance(f, dict) else getattr(f, "na
