# src/processing/__init__.py
# Subpackage initializer for processing pipeline steps.
# IMPORTANT: Code in English.

from __future__ import annotations

__all__ = [
    "normalize_text",
    "slugify",
    "cluster_keywords",
    "classify_intent",
    "propose_hierarchy",
    "design_filters",
    "detect_cannibalization",
    "generate_metadata",
    "score_nodes",
]

# Guarded imports so the package loads even if modules are WIP
try:
    from .normalize import normalize_text, slugify  # type: ignore
except Exception:  # pragma: no cover
    def normalize_text(s: str) -> str:
        raise ImportError("normalize_text not available. Implement src/processing/normalize.py")
    def slugify(s: str) -> str:
        raise ImportError("slugify not available. Implement src/processing/normalize.py")

try:
    from .clustering import cluster_keywords  # type: ignore
except Exception:  # pragma: no cover
    def cluster_keywords(*args, **kwargs):
        raise ImportError("cluster_keywords not available. Implement src/processing/clustering.py")

try:
    from .intent import classify_intent  # type: ignore
except Exception:  # pragma: no cover
    def classify_intent(*args, **kwargs):
        raise ImportError("classify_intent not available. Implement src/processing/intent.py")

try:
    from .hierarchy import propose_hierarchy  # type: ignore
except Exception:  # pragma: no cover
    def propose_hierarchy(*args, **kwargs):
        raise ImportError("propose_hierarchy not available. Implement src/processing/hierarchy.py")

try:
    from .filters import design_filters  # type: ignore
except Exception:  # pragma: no cover
    def design_filters(*args, **kwargs):
        raise ImportError("design_filters not available. Implement src/processing/filters.py")

try:
    from .cannibalization import detect_cannibalization  # type: ignore
except Exception:  # pragma: no cover
    def detect_cannibalization(*args, **kwargs):
        raise ImportError("detect_cannibalization not available. Implement src/processing/cannibalization.py")

try:
    from .metadata import generate_metadata  # type: ignore
except Exception:  # pragma: no cover
    def generate_metadata(*args, **kwargs):
        raise ImportError("generate_metadata not available. Implement src/processing/metadata.py")

try:
    from .scoring import score_nodes  # type: ignore
except Exception:  # pragma: no cover
    def score_nodes(*args, **kwargs):
        raise ImportError("score_nodes not available. Implement src/processing/scoring.py")
