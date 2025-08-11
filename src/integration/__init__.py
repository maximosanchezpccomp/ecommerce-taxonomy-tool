# src/integration/__init__.py
# Subpackage initializer for external integrations (OpenAI, GSC).
# IMPORTANT: Code in English.

from __future__ import annotations

__all__ = ["OpenAIClient", "GSCClient"]

try:
    from .openai_client import OpenAIClient  # type: ignore
except Exception:  # pragma: no cover
    class OpenAIClient:  # minimal stub
        def __init__(self, api_key: str | None = None, model: str | None = None):
            raise ImportError("OpenAIClient not available. Implement src/integration/openai_client.py")

try:
    from .gsc_client import GSCClient  # type: ignore
except Exception:  # pragma: no cover
    class GSCClient:  # minimal stub
        def __init__(self, credentials: dict | None = None):
            raise ImportError("GSCClient not available. Implement src/integration/gsc_client.py")
