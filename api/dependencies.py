"""
api/dependencies.py
===================
FastAPI dependency injection — provides a shared RAGPipeline singleton
across all requests without re-loading models on every call.
"""

from __future__ import annotations

from functools import lru_cache
from pipeline import RAGPipeline


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    """
    Create and cache the RAGPipeline singleton.
    Models (SentenceTransformer, CrossEncoder) are loaded once here
    and reused for every API request.
    """
    return RAGPipeline()
