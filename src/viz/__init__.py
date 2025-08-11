# src/viz/__init__.py
# Subpackage initializer for visualization helpers.
# IMPORTANT: Code in English.

from __future__ import annotations

__all__ = ["render_tree_treemap", "render_tree_graph", "render_table"]

try:
    from .tree_graph import render_tree_treemap, render_tree_graph  # type: ignore
except Exception:  # pragma: no cover
    def render_tree_treemap(*args, **kwargs):
        raise ImportError("render_tree_treemap not available. Implement src/viz/tree_graph.py")
    def render_tree_graph(*args, **kwargs):
        raise ImportError("render_tree_graph not available. Implement src/viz/tree_graph.py")

try:
    from .tables import render_table  # type: ignore
except Exception:  # pragma: no cover
    def render_table(*args, **kwargs):
        raise ImportError("render_table not available. Implement src/viz/tables.py")
