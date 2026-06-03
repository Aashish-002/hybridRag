"""
server.py
=========
FastAPI application entry-point.

Run:
    python server.py
    -- or --
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

Interactive docs (auto-generated):
    Swagger UI  → http://localhost:8000/docs
    ReDoc       → http://localhost:8000/redoc
    OpenAPI JSON → http://localhost:8000/openapi.json
"""

from __future__ import annotations

import os
# Windows OMP fix — must be set before any torch/numpy import
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from config.settings import settings
from utils.logger import setup_logging

# Initialise logging
setup_logging(level=settings.log_level)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description="""
## Hybrid RAG — REST API

Upload documents and query them using Hybrid Retrieval-Augmented Generation:
- **Semantic search** (FAISS dense vectors, BAAI/bge-small-en-v1.5)
- **Keyword search** (BM25)
- **Reciprocal Rank Fusion** to merge both result sets
- **CrossEncoder re-ranking** (ms-marco-MiniLM-L-6-v2)
- **LLM generation** via LM Studio (meta-llama-3-8b-instruct)

### Quick start
1. `POST /ingest` — upload a PDF/DOCX/TXT
2. `POST /query`  — ask a question, get a structured JSON response
3. `GET  /status` — check pipeline and GPU status
""",
    contact={"name": "Hybrid RAG API"},
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ---------------------------------------------------------------------------
# CORS — allow all origins (lock this down in production)
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Register routes (all under /  — no version prefix for simplicity)
# ---------------------------------------------------------------------------

app.include_router(router)

# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,          # set True during development
        log_level="info",
    )
