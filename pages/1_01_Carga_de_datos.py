# pages/1_01_Carga_de_datos.py
# Data loading page (Streamlit multipage)
# IMPORTANT: Code in English. UI labels/messages in Spanish.

import io
import re
import zipfile
import unicodedata
from typing import Dict, Tuple

import pandas as pd
import streamlit as st

# ---------------------------------------
# Try to import shared loaders from src/
# ---------------------------------------
try:
    from src.ingestion.gkp_loader import read_gkp_any  # type: ignore
except Exception:
    read_gkp_any = None

try:
    from src.ingestion.gsc_loader import read_gsc_zip  # type: ignore
except Exception:
    read_gsc_zip = None


# ---------------------------------------
# Local fallbacks (only if src/ not present)
# ---------------------------------------
@st.cache_data(show_spinner=False)
def _fallback_read_gkp_any(file) -> pd.DataFrame:
    """
    Robust GKP parser (CSV/Excel, UTF-16/tab or UTF-8/comma).
    Returns DataFrame with ['keyword', 'avg_monthly_searches'].
    """
    if file is None:
        return pd.DataFrame(columns=["keyword", "avg_monthly_searches"])
    try:
        if file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
        else:
            # try utf-16 + tab, then utf-8 + comma
            file.seek(0)
            try:
                df = pd.read_csv(file, encoding="utf-16", sep="\t")
            except Exception:
                file.seek(0)
                df = pd.read_csv(file, encoding="utf-8", sep=",")
    except Exception:
        file.seek(0)
        df = pd.read_csv(file, engine="python")

    df.columns = [str(c).strip().lower() for c in df.columns]
    kw_col = next(
        (c for c in df.columns if any(k in c for k in ["keyword", "palabra", "term", "termo", "terme", "parola", "consulta"])),
        df.columns[0],
    )
    df.rename(columns={kw_col: "keyword"}, inplace=True)
    ms_col = next(
        (c for c in df.columns if ("avg" in c and "search" in c) or ("promedio" in c) or ("moyenne" in c) or ("média" in c) or ("media" in c and "ricer" in c) or ("monthly" in c and "search" in c)),
        None,
    )
    if ms_col:
        df["avg_monthly_searches"] = pd.to_numeric(
            df[ms_col].astype(str).str.replace('"', "", regex=False).str.replace(".", "", regex=False).str.replace(",", "", regex=False),
            errors="coerce",
        ).fillna(0).astype(int)
    else:
        df["avg_monthly_searches"] = 0

    df["keyword"] = df["keyword"].astype(str).str.strip()
    return df[["keyword", "avg_monthly_searches"]]


@st.cache_data(show_spinner=False)
def _fallback_read_gsc_zip(upload) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load GSC export ZIP: returns (queries_df, pages_df).
    """
    if upload is None:
        return pd.DataFrame(), pd.DataFrame()
    with zipfile.ZipFile(upload) as zf:
        qname = next((n for n in zf.namelist() if any(k in n for k in ["Consulta", "Consultas", "query", "queries"])), None)
        pname = next((n for n in zf.namelist() if any(k in n for k in ["Página", "Páginas", "page", "pages", "urls"])), None)
        qdf = pd.read_csv(zf.open(qname)) if qname else pd.DataFrame()
        pdf = pd.read_csv(zf.open(pname)) if pname else pd.DataFrame()

    if not qdf.empty:
        qdf.columns = [c.strip().lower().replace(" ", "_") for c in qdf.columns]
    if not pdf.empty:
        pdf.columns = [c.strip().lower().replace(" ", "_") for c in pdf.columns]
    return qdf, pdf


# ---------------------------------------
# Utils
# ---------------------------------------
def normalize_text(s: str) -> str:
    s = str(s)
    s = s.lower()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return " ".join(s.split())


@st.cache_data(show_spinner=False)
def dedupe_keywords(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["kw_norm"] = d["keyword"].map(normalize_text)
    # keep max MS for duplicates
    d = d.sort_values("avg_monthly_searches", ascending=False).drop_duplicates(subset=["kw_norm"])
    return d[["keyword", "avg_monthly_searches", "kw_norm"]]


def compute_coverage(per_market: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for mk, d in per_market.items():
        ms = int(d["avg_monthly_searches"].sum()) if not d.empty else 0
        rows.append({"market": mk, "ms_total", "keywords_count"})
        rows[-1]["ms_total"] = ms
        rows[-1]["keywords_count"] = int(d.shape[0])
    return pd.DataFrame(rows)


# ---------------------------------------
# Page UI
# ---------------------------------------
st.header("1) Carga de datos")
st.caption("Mínimo: al menos un CSV/Excel de Keyword Planner (GKP). El resto de inputs son opcionales para mejorar la calidad.")

with st.expander("Instrucciones rápidas"):
    st.markdown(
        """
- **GKP**: puedes cargar uno por mercado (ES/PT/FR/IT). El sistema detecta automáticamente la codificación (UTF-16/UTF-8) y separadores (tab/coma).
- **GSC (ZIP)**: export estándar de Search Console (Consultas y Páginas). Se usa para analizar **gaps** y **canibalización**.
- **Competidores**: CSV/Excel con `url`, `path`, `title` (opcional).
- **Catálogo**: CSV/Excel con `sku`, `brand`, `name`, `attributes...` (opcional).
        """
    )

c1, c2, c3, c4 = st.columns(4)
with c1:
    f_es = st.file_uploader("GKP España (CSV/Excel)", type=["csv", "xlsx"], key="gkp_es")
with c2:
    f_pt = st.file_uploader("GKP Portugal (CSV/Excel)", type=["csv", "xlsx"], key="gkp_pt")
with c3:
    f_fr = st.file_uploader("GKP Francia (CSV/Excel)", type=["csv", "xlsx"], key="gkp_fr")
with c4:
    f_it = st.file_uploader("GKP Italia (CSV/Excel)", type=["csv", "xlsx"], key="gkp_it")

oc1, oc2, oc3 = st.columns(3)
with oc1:
    f_comp = st.file_uploader("Estructuras/paths competidores (CSV/Excel)", type=["csv", "xlsx"], key="comp_paths")
with oc2:
    f_cat = st.file_uploader("Catálogo interno (CSV/Excel)", type=["csv", "xlsx"], key="catalog")
with oc3:
    f_gsc = st.file_uploader("Google Search Console (ZIP)", type=["zip"], key="gsc_zip")

st.divider()
process = st.button("Procesar y previsualizar")

if process:
    with st.spinner("Procesando ficheros..."):
        # Use shared loaders if available; otherwise fallback
        _read_gkp = read_gkp_any or _fallback_read_gkp_any
        _read_gsc = read_gsc_zip or _fallback_read_gsc_zip

        per_market: Dict[str, pd.DataFrame] = {}

        for mk, f in {"es": f_es, "pt": f_pt, "fr": f_fr, "it": f_it}.items():
            if f is None:
                continue
            try:
                df0 = _read_gkp(f)
                df = dedupe_keywords(df0)
                per_market[mk] = df
            except Exception as e:
                st.error(f"[{mk.upper()}] Error leyendo GKP: {e}")

        if not per_market:
            st.error("Debes cargar al menos un GKP para continuar.")
            st.stop()

        # Optional: competitors
        comp_df = pd.DataFrame()
        if f_comp is not None:
            try:
                comp_df = pd.read_csv(f_comp) if f_comp.name.endswith(".csv") else pd.read_excel(f_comp)
            except Exception as e:
                st.warning(f"No se pudo leer el archivo de competidores: {e}")

        # Optional: catalog
        cat_df = pd.DataFrame()
        if f_cat is not None:
            try:
                cat_df = pd.read_csv(f_cat) if f_cat.name.endswith(".csv") else pd.read_excel(f_cat)
            except Exception as e:
                st.warning(f"No se pudo leer el archivo de catálogo: {e}")

        # Optional: GSC
        qdf, pdf = pd.DataFrame(), pd.DataFrame()
        if f_gsc is not None:
            try:
                qdf, pdf = _read_gsc(f_gsc)
            except Exception as e:
                st.warning(f"No se pudo leer el ZIP de GSC: {e}")

        # Save to session
        st.session_state.setdefault("gkp_data", {})
        st.session_state["gkp_data"].update(per_market)
        st.session_state["competitors"] = comp_df
        st.session_state["catalog"] = cat_df
        st.session_state["gsc_queries"] = qdf
        st.session_state["gsc_pages"] = pdf

        # Coverage
        coverage_df = compute_coverage(per_market)
        st.session_state["coverage_by_market"] = coverage_df

        st.success("Datos cargados y normalizados. Previews abajo.")

    # --------- Previews ---------
    st.subheader("Cobertura por mercado (GKP)")
    if "coverage_by_market" in st.session_state and not st.session_state["coverage_by_market"].empty:
        cov = st.session_state["coverage_by_market"]
        st.dataframe(cov, use_container_width=True)
        st.download_button(
            "Descargar cobertura (CSV)",
            data=cov.to_csv(index=False).encode("utf-8"),
            file_name="coverage_by_market.csv",
            mime="text/csv",
        )

    st.subheader("Previsualización de GKP (por mercado)")
    for mk, df in st.session_state.get("gkp_data", {}).items():
        st.markdown(f"**{mk.upper()}** — {len(df):,} keywords (únicas normalizadas)")
        st.dataframe(df.head(50), use_container_width=True)

    st.subheader("GSC (opcional)")
    if isinstance(st.session_state.get("gsc_queries"), pd.DataFrame) and not st.session_state["gsc_queries"].empty:
        qdf = st.session_state["gsc_queries"]
        qcol = next((c for c in qdf.columns if "consulta" in c or "query" in c), qdf.columns[0])
        imp = next((c for c in ["impresiones", "impressions"] if c in qdf.columns), None)
        top = qdf.sort_values(imp, ascending=False).head(30) if imp else qdf.head(30)
        st.dataframe(top, use_container_width=True)
        st.download_button(
            "Descargar consultas GSC (CSV)",
            data=qdf.to_csv(index=False).encode("utf-8"),
            file_name="gsc_queries.csv",
            mime="text/csv",
        )
    else:
        st.caption("No se han cargado consultas de GSC.")

    if isinstance(st.session_state.get("gsc_pages"), pd.DataFrame) and not st.session_state["gsc_pages"].empty:
        st.dataframe(st.session_state["gsc_pages"].head(30), use_container_width=True)
        st.download_button(
            "Descargar páginas GSC (CSV)",
            data=st.session_state["gsc_pages"].to_csv(index=False).encode("utf-8"),
            file_name="gsc_pages.csv",
            mime="text/csv",
        )

    st.subheader("Competidores (opcional)")
    comp_df = st.session_state.get("competitors", pd.DataFrame())
    if not comp_df.empty:
        st.dataframe(comp_df.head(50), use_container_width=True)

    st.subheader("Catálogo (opcional)")
    cat_df = st.session_state.get("catalog", pd.DataFrame())
    if not cat_df.empty:
        st.dataframe(cat_df.head(50), use_container_width=True)

st.divider()
col_a, col_b = st.columns(2)
with col_a:
    if st.button("Limpiar datos cargados"):
        for k in ["gkp_data", "coverage_by_market", "gsc_queries", "gsc_pages", "competitors", "catalog"]:
            if k in st.session_state:
                del st.session_state[k]
        st.experimental_rerun()
with col_b:
    st.caption("Después de cargar, continúa con **2) Parámetros** o **2) Ejecutar** en la página principal.")
