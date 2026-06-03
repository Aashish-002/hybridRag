"""
api/routes.py
=============
All FastAPI route handlers for the Hybrid RAG API.

Endpoints:
    GET  /                    Health check
    GET  /status              Pipeline + GPU + index status
    GET  /documents           List indexed documents
    POST /ingest              Upload and index a document
    POST /query               Ask a question (full JSON response)
    POST /query/stream        Ask a question (Server-Sent Events streaming)
    DELETE /memory            Clear conversation memory
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import AsyncIterator, List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from api.dependencies import get_pipeline
from api.schemas import (
    ChunkDetail,
    DocumentInfo,
    DocumentsResponse,
    ErrorResponse,
    GPUInfo,
    IndexStats,
    IngestResponse,
    LatencyBreakdown,
    MemoryResponse,
    QueryRequest,
    QueryResponse,
    StatusResponse,
)
from config.settings import settings
from pipeline import RAGPipeline
from utils.helpers import extract_file_metadata, get_gpu_info, human_readable_size
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_chunk_detail(doc, score: float) -> ChunkDetail:
    """Convert a (Document, score) pair to a ChunkDetail schema."""
    m = doc.metadata
    return ChunkDetail(
        chunk_id=m.get("chunk_id"),
        source=m.get("source", "unknown"),
        page=m.get("page"),
        content=doc.page_content,
        semantic_score=m.get("semantic_score"),
        bm25_score=m.get("bm25_score"),
        fused_score=m.get("fused_score"),
        rerank_score=m.get("rerank_score"),
    )


def _allowed_file(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in settings.allowed_extensions


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

@router.get("/", tags=["Health"], summary="Health check")
def root():
    """Returns a simple alive signal."""
    return {"status": "ok", "service": settings.app_title, "version": settings.app_version}


# ---------------------------------------------------------------------------
# GET /status
# ---------------------------------------------------------------------------

@router.get(
    "/status",
    response_model=StatusResponse,
    tags=["System"],
    summary="Pipeline, GPU, and index status",
)
def get_status(pipeline: RAGPipeline = Depends(get_pipeline)):
    """
    Returns:
    - Whether the pipeline has indexed documents
    - FAISS vector count and dimension
    - GPU VRAM usage
    - Current LLM and embedding configuration
    """
    index_stats = pipeline.retriever.faiss_store.get_stats()
    gpu = get_gpu_info()
    indexed_docs = len(list(Path(settings.documents_dir).glob("*.*"))) \
        if Path(settings.documents_dir).exists() else 0

    return StatusResponse(
        status="ready" if pipeline.is_ready() else "empty — upload documents first",
        indexed_documents=indexed_docs,
        index_stats=IndexStats(**index_stats),
        gpu=GPUInfo(
            available=gpu.get("available", False),
            name=gpu.get("name", "CPU only"),
            total_mb=gpu.get("total_mb", 0),
            reserved_mb=gpu.get("reserved_mb", 0),
            free_mb=gpu.get("free_mb", 0),
        ),
        llm_provider=settings.llm.provider,
        llm_model=settings.llm.model_name,
        llm_base_url=settings.llm.base_url,
        embed_model=settings.embedding.model_name,
        embed_device=settings.embedding.device,
    )


# ---------------------------------------------------------------------------
# GET /documents
# ---------------------------------------------------------------------------

@router.get(
    "/documents",
    response_model=DocumentsResponse,
    tags=["Documents"],
    summary="List all indexed documents",
)
def list_documents():
    """Returns the list of files that have been uploaded and indexed."""
    doc_dir = Path(settings.documents_dir)
    if not doc_dir.exists():
        return DocumentsResponse(status="ok", count=0, documents=[])

    docs: List[DocumentInfo] = []
    for f in sorted(doc_dir.iterdir()):
        if f.suffix.lower() in settings.allowed_extensions:
            docs.append(DocumentInfo(
                filename=f.name,
                size_human=human_readable_size(f.stat().st_size),
                extension=f.suffix.lower(),
            ))

    return DocumentsResponse(status="ok", count=len(docs), documents=docs)


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------

@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"],
    summary="Upload and index a document",
    responses={
        400: {"model": ErrorResponse, "description": "Unsupported file type"},
        413: {"model": ErrorResponse, "description": "File too large"},
        500: {"model": ErrorResponse, "description": "Ingestion failed"},
    },
)
async def ingest_document(
    file: UploadFile = File(..., description="PDF, DOCX, or TXT file to index"),
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """
    Upload a document and index it into both FAISS and BM25.

    **Postman setup:**
    - Method: POST
    - URL: http://localhost:8000/ingest
    - Body → form-data → Key: `file` (type: File) → select your PDF/DOCX/TXT
    """
    # Validate extension
    if not _allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{Path(file.filename).suffix}'. "
                   f"Allowed: {settings.allowed_extensions}",
        )

    # Validate size
    content = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content) / 1024 / 1024:.1f} MB). "
                   f"Max allowed: {settings.max_upload_size_mb} MB.",
        )

    try:
        result = pipeline.ingest(file_bytes=content, filename=file.filename)
        logger.info("Ingested '%s' via API.", file.filename)
        return IngestResponse(**result)
    except Exception as exc:
        logger.error("Ingestion failed for '%s': %s", file.filename, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

@router.post(
    "/query",
    response_model=QueryResponse,
    tags=["RAG"],
    summary="Ask a question — full JSON response",
    responses={
        400: {"model": ErrorResponse, "description": "No documents indexed"},
        500: {"model": ErrorResponse, "description": "Query failed"},
    },
)
def query(
    body: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """
    Run the full Hybrid RAG pipeline and return a structured JSON response.

    **Response includes:**
    - `answer` — LLM-generated answer
    - `sources` — cited document names and page numbers
    - `chunks` — all retrieved + re-ranked chunks with individual scores
    - `latency` — timing breakdown (retrieval / rerank / llm / total ms)

    **Postman setup:**
    - Method: POST
    - URL: http://localhost:8000/query
    - Body → raw → JSON:
    ```json
    {
      "question": "What are the main findings?",
      "use_memory": true
    }
    ```
    """
    if not pipeline.is_ready():
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. POST to /ingest first.",
        )

    # Apply per-request overrides to settings
    if body.top_k_semantic:
        settings.retrieval.top_k_semantic = body.top_k_semantic
    if body.top_k_bm25:
        settings.retrieval.top_k_bm25 = body.top_k_bm25
    if body.top_k_rerank:
        settings.retrieval.top_k_rerank = body.top_k_rerank

    try:
        result = pipeline.query(
            question=body.question,
            use_memory=body.use_memory,
            stream=False,
        )
    except Exception as exc:
        logger.error("Query failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    chunks = [
        _build_chunk_detail(doc, score)
        for doc, score in result.get("chunks", [])
    ]
    meta = result.get("retrieval_meta", {})

    return QueryResponse(
        question=body.question,
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        chunks=chunks,
        latency=LatencyBreakdown(
            retrieval_ms=meta.get("retrieval_ms", 0),
            rerank_ms=meta.get("rerank_ms", 0),
            llm_ms=meta.get("llm_ms", 0),
            total_ms=meta.get("total_ms", 0),
            num_retrieved=meta.get("num_retrieved", 0),
            num_reranked=meta.get("num_reranked", 0),
        ),
    )


# ---------------------------------------------------------------------------
# POST /query/stream   — Server-Sent Events
# ---------------------------------------------------------------------------

@router.post(
    "/query/stream",
    tags=["RAG"],
    summary="Ask a question — streaming tokens via SSE",
    response_description="Server-Sent Events stream of answer tokens",
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def query_stream(
    body: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """
    Stream the LLM answer token-by-token using Server-Sent Events.

    Each event is:  `data: <token>\\n\\n`
    Final event is: `data: [DONE]\\n\\n`

    **Postman setup:**
    - Method: POST
    - URL: http://localhost:8000/query/stream
    - Body → raw → JSON: `{"question": "Summarize the document"}`
    - Send → watch the response body fill in real time

    **Note:** Postman will buffer SSE and show all at once when complete.
    For true streaming, use curl:
    ```
    curl -X POST http://localhost:8000/query/stream \\
         -H "Content-Type: application/json" \\
         -d "{\"question\": \"What is this document about?\"}" \\
         --no-buffer
    ```
    """
    if not pipeline.is_ready():
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. POST to /ingest first.",
        )

    def token_generator():
        try:
            for token in pipeline.query_stream(body.question):
                # SSE format: data: <payload>\n\n
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.error("Streaming query failed: %s", exc, exc_info=True)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",        # disable Nginx buffering
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# DELETE /memory
# ---------------------------------------------------------------------------

@router.delete(
    "/memory",
    response_model=MemoryResponse,
    tags=["RAG"],
    summary="Clear conversation memory",
)
def clear_memory(pipeline: RAGPipeline = Depends(get_pipeline)):
    """
    Clears the conversation history so the next question starts fresh.

    **Postman setup:**
    - Method: DELETE
    - URL: http://localhost:8000/memory
    """
    pipeline.clear_memory()
    return MemoryResponse()
