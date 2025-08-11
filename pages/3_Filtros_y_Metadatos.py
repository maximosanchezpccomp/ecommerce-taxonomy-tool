# pages/3_Filtros_y_Metadatos.py
# Filters & Metadata page (Streamlit multipage)
# IMPORTANT: Code in English. UI labels/messages in Spanish.

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

# -------------------------------
# Data models (mirroring main app)
# -------------------------------
@dataclass
class SEOData:
    title: str
    h1: str
    meta: str
    faq: List[str] = field(default_factory=list)

@dataclass
class FilterDef:
    name: str
    type: str  # "in", "range", "boolean"
    values: List[str] = field(default_factory=list)
    order: int = 0
    visibility: str = "core"  # "core" | "advanced" | "hidden_mobile"

# -------------------------------
# Helpers
# -------------------------------
def normalize_text(s: str) -> str:
    s = str(s)
    s = s.lower()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return " ".join(s.split())

def count_chars(text: str) -> int:
    return len(text or "")

def within_range(n: int, lo: int, hi: int) -> bool:
    return lo <= n <= hi

def json_safe(obj) -> dict:
    # Convert dataclasses & lists of dataclasses to JSON-safe dicts
    if isinstance(obj, (SEOData, FilterDef)):
        return asdict(obj)
    if isinstance(obj, list):
        return [json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    return obj

def coerce_filters(df: pd.DataFrame) -> List[FilterDef]:
    out: List[FilterDef] = []
    if df.empty:
        return out
    for _, r in df.iterrows():
        name = str(r.get("name", "")).strip()
        ftype = str(r.get("type", "in")).strip().lower()
        values_raw = str(r.get("values", "") or "")
        values = [v.strip() for v in re.split(r"[|,;/]", values_raw) if v.strip()] if ftype == "in" else []
        order = int(r.get("order", 0) or 0)
        visibility = str(r.get("visibility", "core")).strip().lower()
        out.append(FilterDef(name=name, type=ftype, values=values, order=order, visibility=visibility))
    return out

def suggest_filters_from_catalog(catalog: pd.DataFrame) -> List[FilterDef]:
    """
    Heuristic suggestions from a generic catalog:
    - If 'brand'/'marca' exists -> Brand filter (in) with top values
    - Always include Price range (range) as safe default
    - Scan common attribute columns and propose 'in' filters with top distinct values
    """
    suggestions: List[FilterDef] = []

    # Brand
    brand_cols = [c for c in catalog.columns if c.lower() in ["brand", "marca"]]
    if brand_cols:
        col = brand_cols[0]
        top_vals = (
            catalog[col]
            .dropna()
            .astype(str)
            .str.strip()
            .value_counts()
            .head(12)
            .index.tolist()
        )
        suggestions.append(FilterDef(name="Marca", type="in", values=top_vals, order=1, visibility="core"))

    # Price (generic range)
    suggestions.append(FilterDef(name="Precio", type="range", values=[], order=2, visibility="core"))

    # Attribute columns (heuristics)
    attr_patterns = {
        "Resolución": ["resol", "resolution"],
        "Conectividad": ["conect", "connect", "wifi", "ethernet", "bluetooth", "poe"],
        "Almacenamiento": ["almacen", "storage", "sd", "micro", "nube", "cloud"],
        "Alimentación": ["alimenta", "power", "bateria", "battery", "cable"],
        "Tamaño / Diagonal": ["tamaño", "tamano", "diagonal", "pulg", "inch"],
        "Frecuencia / Hz": ["hz", "frecuen", "frequency", "refresh"],
        "Memoria / RAM": ["ram", "memoria"],
        "Capacidad": ["capacidad", "capacity", "gb", "tb"],
        "Panel / Tipo": ["panel", "tipo", "type", "oled", "ips", "tn", "va"],
    }

    order = 3
    for fname, pats in attr_patterns.items():
        cands = [c for c in catalog.columns if any(p in c.lower() for p in pats)]
        if not cands:
            continue
        # Choose the first matching column with categorical-ish values
        col = cands[0]
        # Build value list
        vals = (
            catalog[col]
            .dropna()
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
        # If it looks numeric, skip as range (handled by price)
        if pd.to_numeric(vals, errors="coerce").notna().mean() > 0.7:
            continue
        top_vals = vals.value_counts().head(12).index.tolist()
        if len(top_vals) >= 2:
            suggestions.append(FilterDef(name=fname, type="in", values=top_vals, order=order, visibility="core"))
            order += 1

    # Add an AI/Detection boolean if likely present
    suggestions.append(FilterDef(name="Detección de movimiento / IA", type="boolean", values=[], order=order, visibility="advanced"))

    return suggestions

def generate_metadata_template(node_name: str, market: str) -> SEOData:
    titles = {
        "es": f"{node_name} al mejor precio | Envío rápido",
        "pt": f"{node_name} ao melhor preço | Envio rápido",
        "fr": f"{node_name} au meilleur prix | Expédition rapide",
        "it": f"{node_name} al miglior prezzo | Spedizione rapida",
    }
    metas = {
        "es": f"Compra {node_name} con envío rápido y garantía. Filtra por marca, precio y especificaciones.",
        "pt": f"Compre {node_name} com envio rápido e garantia. Filtre por marca, preço e especificações.",
        "fr": f"Achetez {node_name} avec expédition rapide et garantie. Filtrez par marque, prix et spécifications.",
        "it": f"Acquista {node_name} con spedizione rapida e garanzia. Filtra per marca, prezzo e specifiche.",
    }
    faqs = {
        "es": [
            "¿Qué debo comparar antes de comprar?",
            "¿Cómo filtrar por marca, precio y especificaciones?",
            "¿Compatibilidad con asistentes de voz y apps?"
        ],
        "pt": [
            "O que comparar antes de comprar?",
            "Como filtrar por marca, preço e especificações?",
            "Compatibilidade com assistentes de voz e apps?"
        ],
        "fr": [
            "Que comparer avant d’acheter ?",
            "Comment filtrer par marque, prix et spécifications ?",
            "Compatibilité avec assistants vocaux et applications ?"
        ],
        "it": [
            "Cosa confrontare prima di acquistare?",
            "Come filtrare per marca, prezzo e specifiche?",
            "Compatibilità con assistenti vocali e app?"
        ],
    }
    t = titles.get(market, titles["es"])[:65]
    m = metas.get(market, metas["es"])[:160]
    return SEOData(title=t, h1=node_name, meta=m, faq=faqs.get(market, faqs["es"]))

def validate_seo_lengths(seo: SEOData) -> Dict[str, Dict[str, str]]:
    """
    Validate SEO lengths. Target ranges:
      - title: 60–65 chars
      - meta: 140–160 chars
    """
    report = {}
    tl = count_chars(seo.title)
    ml = count_chars(seo.meta)
    report["title"] = {
        "length": str(tl),
        "status": "ok" if within_range(tl, 60, 65) else ("warn" if within_range(tl, 56, 70) else "bad"),
        "hint": "60–65" }
    report["meta"] = {
        "length": str(ml),
        "status": "ok" if within_range(ml, 140, 160) else ("warn" if within_range(ml, 130, 170) else "bad"),
        "hint": "140–160" }
    return report

def status_badge(status: str) -> str:
    colors = {"ok": "✅", "warn": "🟡", "bad": "🔴"}
    return colors.get(status, "⚪")

# -------------------------------
# Preconditions
# -------------------------------
st.header("4) Filtros y Metadatos")
st.caption("Define filtros por subcategoría y genera metadatos SEO (Title, H1, Meta, FAQ).")

if "taxonomy_nodes" not in st.session_state or not st.session_state["taxonomy_nodes"]:
    st.error("No hay una jerarquía propuesta en sesión. Ve a **3) Clustering y Taxonomía** y genera la jerarquía.")
    st.stop()

params = st.session_state.get("params", {})
markets: List[str] = params.get("markets", list(st.session_state["taxonomy_nodes"].keys()))
available_markets = sorted([mk for mk in st.session_state["taxonomy_nodes"].keys() if mk in markets])

if not available_markets:
    st.error("No hay mercados disponibles para edición.")
    st.stop()

market = st.selectbox("Selecciona mercado", available_markets, index=0)
nodes = st.session_state["taxonomy_nodes"][market]

# Choose child node to edit (skip the root)
children = [n for n in nodes if n.get("parent_id")]
if not children:
    st.warning("No hay subcategorías para editar en este mercado.")
    st.stop()

node_names = [n["name"] for n in children]
node_idx = st.selectbox("Selecciona subcategoría", list(range(len(node_names))), format_func=lambda i: node_names[i])
current = children[int(node_idx)]

tabs = st.tabs(["🎛️ Filtros", "📝 Metadatos"])

# -------------------------------
# Tab: Filters
# -------------------------------
with tabs[0]:
    st.subheader(f"Filtros — {current['name']}")
    # Existing filters as editable data editor
    existing = current.get("filters", [])

    df_filters = pd.DataFrame(
        [{"name": f.get("name") if isinstance(f, dict) else f.name,
          "type": f.get("type") if isinstance(f, dict) else f.type,
          "values": "|".join((f.get("values") if isinstance(f, dict) else f.values) or []),
          "order": f.get("order") if isinstance(f, dict) else f.order,
          "visibility": f.get("visibility") if isinstance(f, dict) else f.visibility}
         for f in (existing or [])]
    ) if existing else pd.DataFrame(columns=["name","type","values","order","visibility"])

    st.caption("Edita las celdas. `values` acepta lista separada por `|` para filtros tipo **in**.")
    edited = st.data_editor(
        df_filters,
        num_rows="dynamic",
        column_config={
            "type": st.column_config.SelectboxColumn("type", options=["in","range","boolean"], default="in"),
            "visibility": st.column_config.SelectboxColumn("visibility", options=["core","advanced","hidden_mobile"], default="core"),
            "order": st.column_config.NumberColumn("order", min_value=0, max_value=99, step=1),
        },
        use_container_width=True
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Añadir fila básica (Marca | Precio | Atributo)"):
            base = pd.DataFrame([
                {"name": "Marca", "type": "in", "values": "", "order": 1, "visibility": "core"},
                {"name": "Precio", "type": "range", "values": "", "order": 2, "visibility": "core"},
                {"name": "Atributo 1", "type": "in", "values": "", "order": 3, "visibility": "core"},
            ])
            edited = pd.concat([edited, base], ignore_index=True)
    with c2:
        if st.button("Sugerir desde catálogo (si disponible)"):
            catalog: pd.DataFrame = st.session_state.get("catalog", pd.DataFrame())
            if catalog is None or catalog.empty:
                st.warning("No hay catálogo cargado en sesión.")
            else:
                sugg = suggest_filters_from_catalog(catalog)
                df_sugg = pd.DataFrame([{
                    "name": s.name, "type": s.type, "values": "|".join(s.values), "order": s.order, "visibility": s.visibility
                } for s in sugg])
                edited = pd.concat([edited, df_sugg], ignore_index=True).drop_duplicates(subset=["name"], keep="first")
    with c3:
        if st.button("Ordenar por 'order' y limpiar duplicados"):
            edited = edited.drop_duplicates(subset=["name"], keep="first").sort_values("order", ascending=True).reset_index(drop=True)

    st.divider()
    if st.button("Guardar filtros en la sesión"):
        # Coerce and save back to node
        current["filters"] = json_safe(coerce_filters(edited))
        # Propagate to session
        # Replace the edited node in the market list
        for i, n in enumerate(nodes):
            if n["id"] == current["id"]:
                nodes[i] = current
                break
        st.session_state["taxonomy_nodes"][market] = nodes
        st.success("Filtros guardados para esta subcategoría.")

# -------------------------------
# Tab: Metadata
# -------------------------------
with tabs[1]:
    st.subheader(f"Metadatos SEO — {current['name']}")
    # Pull existing SEO or create defaults
    existing_seo = current.get("seo", {}) or {}
    seo = SEOData(
        title=existing_seo.get("title", ""),
        h1=existing_seo.get("h1", current["name"]),
        meta=existing_seo.get("meta", ""),
        faq=existing_seo.get("faq", []),
    )

    c1, c2 = st.columns([2,1])
    with c1:
        title = st.text_input("Title (60–65 chars)", value=seo.title, max_chars=80)
        h1 = st.text_input("H1", value=seo.h1, max_chars=120)
        meta = st.text_area("Meta description (140–160 chars)", value=seo.meta, height=120, max_chars=220)
    with c2:
        st.caption("Ayuda rápida")
        if st.button("Rellenar plantilla por idioma (sin LLM)"):
            market_lang = market  # es/pt/fr/it
            tpl = generate_metadata_template(current["name"], market_lang)
            title, h1, meta = tpl.title, tpl.h1, tpl.meta
            seo = tpl

    # FAQ editor
    st.markdown("**FAQs (3–5 sugeridas)**")
    faqs_df = pd.DataFrame({"faq": seo.faq or [""]})
    faqs_edit = st.data_editor(faqs_df, num_rows="dynamic", use_container_width=True)
    faqs_list = [str(x).strip() for x in faqs_edit["faq"].tolist() if str(x).strip()]

    # Validation badges
    report = validate_seo_lengths(SEOData(title=title, h1=h1, meta=meta, faq=faqs_list))
    st.caption(f"Title: {status_badge(report['title']['status'])} {report['title']['length']} chars (objetivo {report['title']['hint']})")
    st.caption(f"Meta: {status_badge(report['meta']['status'])} {report['meta']['length']} chars (objetivo {report['meta']['hint']})")

    # Slug (optional quick edit)
    st.markdown("**Slug (opcional)**")
    slug_default = current.get("slug", "")
    new_slug = st.text_input("Slug de la subcategoría", value=slug_default, help="Opcional. Si lo editas aquí, recuerda revisar duplicidades en el mercado.")

    # Save metadata
    st.divider()
    if st.button("Guardar metadatos en la sesión"):
        current["seo"] = json_safe(SEOData(title=title, h1=h1, meta=meta, faq=faqs_list))
        if new_slug and isinstance(new_slug, str):
            current["slug"] = new_slug.strip()
        # Replace node in market list
        for i, n in enumerate(nodes):
            if n["id"] == current["id"]:
                nodes[i] = current
                break
        st.session_state["taxonomy_nodes"][market] = nodes
        st.success("Metadatos guardados para esta subcategoría.")

# -------------------------------
# Summary & quick checks
# -------------------------------
st.divider()
st.subheader("Resumen rápido del mercado")
summary_rows = []
for n in st.session_state["taxonomy_nodes"][market]:
    if n.get("parent_id"):
        fcore = [f["name"] if isinstance(f, dict) else getattr(f, "name", "") for f in (n.get("filters") or []) if (isinstance(f, dict) and f.get("visibility") == "core") or (not isinstance(f, dict) and getattr(f, "visibility", "") == "core")]
        seod = n.get("seo", {})
        summary_rows.append({
            "name": n["name"],
            "slug": n.get("slug", ""),
            "core_filters": ", ".join(fcore) if fcore else "",
            "title_len": len((seod.get("title") or "")),
            "meta_len": len((seod.get("meta") or "")),
        })
summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df, use_container_width=True)

# Detect duplicate slugs within market
if not summary_df.empty and "slug" in summary_df.columns:
    dup = summary_df["slug"].value_counts()
    dups = dup[dup > 1]
    if len(dups) > 0:
        st.warning(f"Hay slugs duplicados en este mercado: {', '.join(dups.index.tolist())}. Revisa posibles conflictos y canibalización.")
