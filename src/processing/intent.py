# src/processing/intent.py
# Intent classification (heuristics + optional LLM arbitration) for ES/PT/FR/IT.
# IMPORTANT: Code in English.

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Local helpers
from .normalize import normalize_text


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------
INTENT_LABELS = ("informational", "mixed", "transactional")


@dataclass
class IntentParams:
    """
    Parameters controlling intent classification.
    """
    use_llm_arbiter: bool = True
    rules_weight: float = 0.5  # 0..1  (LLM weight is 1 - rules_weight)
    model: str = "gpt-5-thinking"
    temperature: float = 0.0
    batch_size: int = 40  # how many keywords per LLM batch


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def classify_intent(
    df: pd.DataFrame,
    *,
    text_col: str = "keyword",
    market: str = "es",
    params: Optional[IntentParams] = None,
    openai_client=None,
) -> pd.DataFrame:
    """
    Classify keyword intent into {'informational','mixed','transactional'} using:
      1) Multilingual rule-based heuristics
      2) Optional LLM arbitration (if `params.use_llm_arbiter` and `openai_client` provided)

    Returns a copy of `df` with added columns:
      - intent_rules: heuristic label
      - intent_rules_score: numeric in [0,1] (0=informational, 0.5=mixed, 1=transactional)
      - intent_llm: LLM label (if available)
      - intent_llm_conf: LLM soft score (if available)
      - intent: final blended label
      - intent_score: final blended score in [0,1]
    """
    if df is None or df.empty or text_col not in df.columns:
        return pd.DataFrame(columns=list(df.columns) + ["intent", "intent_score"])

    p = params or IntentParams()
    d = df.copy()
    texts = d[text_col].astype(str).fillna("")

    # 1) Heuristic rules
    rules_labels, rules_scores = _heuristic_batch(texts.tolist(), market=market)

    d["intent_rules"] = rules_labels
    d["intent_rules_score"] = rules_scores

    # 2) Optional LLM arbitration
    llm_labels: List[Optional[str]] = [None] * len(d)
    llm_scores: List[Optional[float]] = [None] * len(d)

    if p.use_llm_arbiter and openai_client is not None:
        try:
            llm_labels, llm_scores = _llm_arbiter_batch(
                texts.tolist(),
                market=market,
                client=openai_client,
                model=p.model,
                temperature=p.temperature,
                batch_size=p.batch_size,
            )
        except Exception:
            # Fail silent → keep None, fall back to rules only
            pass

    # 3) Blend
    final_scores: List[float] = []
    final_labels: List[str] = []

    for rlab, rsc, llab, lsc in zip(rules_labels, rules_scores, llm_labels, llm_scores):
        # map label->score if score missing
        if lsc is None and llab is not None:
            lsc = _label_to_score(llab)
        if lsc is None:
            lsc = rsc  # fallback

        w_rules = float(p.rules_weight)
        w_llm = 1.0 - w_rules

        score = w_rules * float(rsc) + w_llm * float(lsc)
        label = _score_to_label(score)
        final_scores.append(float(score))
        final_labels.append(label)

    d["intent_llm"] = llm_labels
    d["intent_llm_conf"] = llm_scores
    d["intent_score"] = final_scores
    d["intent"] = final_labels
    return d


# ---------------------------------------------------------------------
# Heuristic rules (multilingual)
# ---------------------------------------------------------------------
def _heuristic_batch(texts: List[str], market: str = "es") -> Tuple[List[str], List[float]]:
    labels: List[str] = []
    scores: List[float] = []
    for t in texts:
        lab, sc = _heuristic_one(t, market=market)
        labels.append(lab)
        scores.append(sc)
    return labels, scores


def _heuristic_one(text: str, market: str = "es") -> Tuple[str, float]:
    """
    Very simple, transparent rules. Returns (label, score in [0,1]).
    """
    t = normalize_text(text)
    lang = (market or "es").lower()

    # Transactional triggers per language
    tx_terms = {
        "es": r"(comprar|precio|precios|oferta|ofertas|barat[oa]s?|rebajas|financiaci[oó]n|env[ií]o|tienda|stock)",
        "pt": r"(comprar|pre[cç]o|oferta|promo[cç][aã]o|barat[oa]s?|envio|loja|em stock)",
        "fr": r"(acheter|prix|promo|r[eé]duction|pas cher|livraison|magasin|en stock)",
        "it": r"(comprare|prezzo|offerta|sconto|economico|spedizione|negozio|disponibile)",
    }
    # Informational triggers
    info_terms = {
        "es": r"(c[oó]mo|que es|qu[eé] es|gu[ií]a|opiniones|review|comparativa|vs|diferencia|mejores\s+\w+)",
        "pt": r"(como|o que [eé]|guia|opini[oõ]es|review|comparativo|vs|diferen[cç]a|melhores\s+\w+)",
        "fr": r"(comment|qu[’']?est[- ]ce que|guide|avis|test|comparatif|vs|diff[eé]rence|meilleurs\s+\w+)",
        "it": r"(come|che cos[’']?[eé]|guida|recensioni|review|confronto|vs|differenza|migliori\s+\w+)",
    }

    # Model/attribute patterns often signal shopping intent ("mixed" bias)
    spec_pat = r"(\b\d{2,4}\s?(hz|w|mm|cm|mah|mp|gb|tb|inch|\"|\')\b|rtx|gtx|i[3579]-?\d{3,5}|m\.?2|nvme|144hz|2k|4k|1080p|wifi|poe)"
    is_tx = bool(re.search(tx_terms.get(lang, tx_terms["es"]), t))
    is_info = bool(re.search(info_terms.get(lang, info_terms["es"]), t))
    has_specs = bool(re.search(spec_pat, t))

    # Scoring logic (0=info, 0.5=mixed, 1=tx)
    if is_tx and not is_info:
        return "transactional", 1.0
    if is_info and not is_tx:
        # If specs show, nudge to mixed
        return ("mixed", 0.5) if has_specs else ("informational", 0.0)
    if is_tx and is_info:
        return "mixed", 0.6
    # Neutral: rely on specs signal
    if has_specs:
        return "mixed", 0.55

    # Weak default
    return "mixed", 0.5


def _label_to_score(label: Optional[str]) -> float:
    if not label:
        return 0.5
    l = label.strip().lower()
    if l.startswith("trans"):
        return 1.0
    if l.startswith("info"):
        return 0.0
    return 0.5


def _score_to_label(score: float) -> str:
    if score >= 2.0 / 3.0:
        return "transactional"
    if score <= 1.0 / 3.0:
        return "informational"
    return "mixed"


# ---------------------------------------------------------------------
# LLM arbitration (optional)
# ---------------------------------------------------------------------
_LLM_SYSTEM = (
    "You are an SEO assistant. Classify each query into exactly one of: "
    "'informational', 'mixed', or 'transactional'.\n"
    "Definitions:\n"
    "- informational: the user seeks information/comparison/guide, not ready to buy.\n"
    "- mixed: shopping-oriented research (brand/specs/best-of) but not explicit purchase intent.\n"
    "- transactional: clear buy/commercial intent (price, buy, offers, store, delivery).\n"
    "Return a compact JSON object with a list of items like: "
    "{\"q\":\"<query>\", \"intent\":\"transactional\", \"confidence\":0.90}\n"
    "Language can be ES/PT/FR/IT; base your judgment on the query language."
)

_LLM_USER_TEMPLATE = (
    "Classify the following queries (one per line). ONLY return JSON with a 'items' array.\n"
    "Queries:\n{payload}"
)


def _llm_arbiter_batch(
    texts: List[str],
    *,
    market: str,
    client,
    model: str,
    temperature: float,
    batch_size: int = 40,
) -> Tuple[List[Optional[str]], List[Optional[float]]]:
    """
    Call an OpenAI-like client in batches. The client is expected to provide either:
      - client.classify_intent(queries: List[str], model: str, temperature: float) -> List[dict]
      - or a generic chat method:
            client.chat(system: str, user: str, model: str, temperature: float) -> str (JSON)

    Returns two lists aligned with `texts`: (labels, confidences).
    On any failure, returns (None, None) for the affected items.
    """
    labels: List[Optional[str]] = [None] * len(texts)
    confs: List[Optional[float]] = [None] * len(texts)

    if not texts:
        return labels, confs

    # Try optimized path if the client exposes a bulk classifier
    if hasattr(client, "classify_intent"):
        try:
            results = client.classify_intent(texts, model=model, temperature=temperature)  # type: ignore
            # Expect list of {"q": ..., "intent": ..., "confidence": ...}
            for i, r in enumerate(results or []):
                labels[i] = _sanitize_label(r.get("intent"))
                confs[i] = _clip01(r.get("confidence"))
            return labels, confs
        except Exception:
            # fall through to chat path
            pass

    # Generic chat in batches
    for start in range(0, len(texts), max(1, int(batch_size))):
        chunk = texts[start : start + batch_size]
        payload = "\n".join(f"- {q}" for q in chunk)

        try:
            if hasattr(client, "chat"):
                raw = client.chat(system=_LLM_SYSTEM, user=_LLM_USER_TEMPLATE.format(payload=payload), model=model, temperature=temperature)  # type: ignore
                items = _parse_llm_json_items(raw)
            elif hasattr(client, "complete"):
                raw = client.complete(prompt=_LLM_SYSTEM + "\n\n" + _LLM_USER_TEMPLATE.format(payload=payload), model=model, temperature=temperature)  # type: ignore
                items = _parse_llm_json_items(raw)
            else:
                # Unknown client interface
                items = []
        except Exception:
            items = []

        # Map back in order; tolerate missing/extra items by greedy matching
        for i, q in enumerate(chunk):
            rec = _best_match_item(q, items)
            if rec:
                labels[start + i] = _sanitize_label(rec.get("intent"))
                confs[start + i] = _clip01(rec.get("confidence"))

    return labels, confs


def _sanitize_label(x: Optional[str]) -> Optional[str]:
    if not x:
        return None
    t = str(x).strip().lower()
    if t.startswith("trans"):
        return "transactional"
    if t.startswith("info"):
        return "informational"
    if t.startswith("mix"):
        return "mixed"
    return None


def _clip01(x) -> Optional[float]:
    try:
        v = float(x)
        if v != v:  # NaN
            return None
        return max(0.0, min(1.0, v))
    except Exception:
        return None


def _parse_llm_json_items(raw: str) -> List[dict]:
    import json

    if not raw:
        return []
    # Extract JSON block (be tolerant to leading/trailing text)
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < start:
        return []
    try:
        obj = json.loads(raw[start : end + 1])
        items = obj.get("items") or obj.get("data") or obj.get("results") or []
        if isinstance(items, list):
            # Normalize structure
            norm = []
            for it in items:
                norm.append(
                    {
                        "q": it.get("q") or it.get("query") or it.get("text"),
                        "intent": _sanitize_label(it.get("intent")) or it.get("label"),
                        "confidence": _clip01(it.get("confidence") or it.get("score")),
                    }
                )
            return norm
    except Exception:
        return []
    return []


def _best_match_item(query: str, items: List[dict]) -> Optional[dict]:
    """
    Simple greedy matcher: exact string match on 'q'; else fallback to first item.
    """
    if not items:
        return None
    for it in items:
        if str(it.get("q", "")).strip().lower() == str(query).strip().lower():
            return it
    return items[0]
