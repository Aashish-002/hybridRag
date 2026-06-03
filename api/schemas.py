"""
api/schemas.py
==============
Pydantic request / response models for the FastAPI layer.
Keeping schemas separate from business logic makes them easy to version.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class ChunkDetail(BaseModel):
    """A single retrieved + re-ranked document chunk."""
    chunk_id: Optional[str] = None
    source: str
    page: Optional[Any] = None
    content: str
    semantic_score: Optional[float] = None
    bm25_score: Optional[float] = None
    fused_score: Optional[float] = None
    rerank_score: Optional[float] = None


class LatencyBreakdown(BaseModel):
    retrieval_ms: int = 0
    rerank_ms: int = 0
    llm_ms: int = 0
    total_ms: int = 0
    num_retrieved: int = 0
    num_reranked: int = 0


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------

class IngestResponse(BaseModel):
    status: str = "ok"
    filename: str
    num_pages: int
    num_chunks: int
    elapsed_ms: int


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000,
                          example="What are the key findings of the report?")
    use_memory: bool = Field(default=True,
                             description="Include conversation history in the prompt.")
    top_k_semantic: Optional[int] = Field(default=None, ge=1, le=50)
    top_k_bm25: Optional[int] = Field(default=None, ge=1, le=50)
    top_k_rerank: Optional[int] = Field(default=None, ge=1, le=20)


class QueryResponse(BaseModel):
    status: str = "ok"
    question: str
    answer: str
    sources: List[str]
    chunks: List[ChunkDetail]
    latency: LatencyBreakdown


# ---------------------------------------------------------------------------
# POST /query/stream  (Server-Sent Events — text/event-stream)
# ---------------------------------------------------------------------------
# No response schema needed — SSE sends raw token strings.


# ---------------------------------------------------------------------------
# DELETE /memory
# ---------------------------------------------------------------------------

class MemoryResponse(BaseModel):
    status: str = "ok"
    message: str = "Conversation memory cleared."


# ---------------------------------------------------------------------------
# GET /status
# ---------------------------------------------------------------------------

class IndexStats(BaseModel):
    vectors: int = 0
    dimension: int = 0


class GPUInfo(BaseModel):
    available: bool
    name: str = "CPU only"
    total_mb: int = 0
    reserved_mb: int = 0
    free_mb: int = 0


class StatusResponse(BaseModel):
    status: str
    indexed_documents: int
    index_stats: IndexStats
    gpu: GPUInfo
    llm_provider: str
    llm_model: str
    llm_base_url: str
    embed_model: str
    embed_device: str


# ---------------------------------------------------------------------------
# GET /documents
# ---------------------------------------------------------------------------

class DocumentInfo(BaseModel):
    filename: str
    size_human: str
    extension: str


class DocumentsResponse(BaseModel):
    status: str = "ok"
    count: int
    documents: List[DocumentInfo]


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    status: str = "error"
    detail: str
