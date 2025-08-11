# src/export/__init__.py
# Subpackage initializer for export utilities.
# IMPORTANT: Code in English.

from __future__ import annotations

__all__ = [
    "export_master_json",
    "export_taxonomy_csv",
    "export_filters_csv",
    "export_keywords_to_node_csv",
    "export_markdown",
    "export_pptx",
]

try:
    from .json_export import export_master_json  # type: ignore
except Exception:  # pragma: no cover
    def export_master_json(*args, **kwargs):
        raise ImportError("export_master_json not available. Implement src/export/json_export.py")

try:
    from .csv_export import export_taxonomy_csv, export_filters_csv, export_keywords_to_node_csv  # type: ignore
except Exception:  # pragma: no cover
    def export_taxonomy_csv(*args, **kwargs):
        raise ImportError("export_taxonomy_csv not available. Implement src/export/csv_export.py")
    def export_filters_csv(*args, **kwargs):
        raise ImportError("export_filters_csv not available. Implement src/export/csv_export.py")
    def export_keywords_to_node_csv(*args, **kwargs):
        raise ImportError("export_keywords_to_node_csv not available. Implement src/export/csv_export.py")

try:
    from .markdown_export import export_markdown  # type: ignore
except Exception:  # pragma: no cover
    def export_markdown(*args, **kwargs):
        raise ImportError("export_markdown not available. Implement src/export/markdown_export.py")

try:
    from .pptx_export import export_pptx  # type: ignore
except Exception:  # pragma: no cover
    def export_pptx(*args, **kwargs):
        raise ImportError("export_pptx not available. Implement src/export/pptx_export.py")
