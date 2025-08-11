# src/__init__.py
# Top-level package initializer for the eCommerce Taxonomy Tool.
# IMPORTANT: Code in English.

from __future__ import annotations

from importlib.metadata import version, PackageNotFoundError

__all__ = ["__version__", "get_version", "PACKAGE_NAME"]

PACKAGE_NAME = "ecommerce_taxonomy_tool"

try:
    __version__ = version(PACKAGE_NAME)
except PackageNotFoundError:
    # In local/dev environments without packaging, default to 0.1.0
    __version__ = "0.1.0"


def get_version() -> str:
    """Return package version string."""
    return __version__
