# src/ingestion/__init__.py
# Subpackage initializer for ingestion utilities.
# IMPORTANT: Code in English.

from __future__ import annotations

__all__ = [
    "read_gkp_any",
    "read_gsc_zip",
    "read_competitor_paths",
]

# Optional, guarded imports to avoid hard failures during early scaffolding
try:
    from .gkp_loader import read_gkp_any  # type: ignore
except Exception:  # pragma: no cover
    def read_gkp_any(*args, **kwargs):
        raise ImportError("read_gkp_any is not available. Implement src/ingestion/gkp_loader.py")

try:
    from .gsc_loader import read_gsc_zip  # type: ignore
except Exception:  # pragma: no cover
    def read_gsc_zip(*args, **kwargs):
        raise ImportError("read_gsc_zip is not available. Implement src/ingestion/gsc_loader.py")

try:
    from .competitor_loader import read_competitor_paths  # type: ignore
except Exception:  # pragma: no cover
    def read_competitor_paths(*args, **kwargs):
        raise ImportError("read_competitor_paths is not available. Implement src/ingestion/competitor_loader.py")
