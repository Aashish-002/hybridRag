"""
documents/loader.py
===================
Document ingestion layer: load raw bytes from PDF / DOCX / TXT files
and return clean, metadata-enriched plain text.

Purpose:
    Abstracts away format-specific parsing so the rest of the pipeline
    always receives a normalised (text, metadata) pair.

Workflow:
    1. Detect file type by extension
    2. Dispatch to the appropriate parser (PyMuPDF / python-docx / plain text)
    3. Clean extracted text via utils.clean_text
    4. Attach rich metadata (source, page count, author, etc.)
    5. Return list[Document]  (LangChain Document objects)

Best Practices:
    - Use PyMuPDF (fitz) instead of PyPDF2 — faster, handles complex PDFs
    - Preserve page numbers in metadata for citations
    - Gracefully handle corrupt/unreadable files without crashing

Performance Considerations:
    - PyMuPDF is the fastest pure-Python PDF parser available
    - python-docx is lightweight for DOCX
    - For very large files (>100 MB) consider streaming rather than load-all
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document

from utils.helpers import clean_text, extract_file_metadata
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Per-format parsers
# ---------------------------------------------------------------------------

def _load_pdf(path: Path) -> List[Tuple[str, dict]]:
    """
    Parse a PDF file page-by-page using PyMuPDF (fitz).

    Returns list of (page_text, page_metadata) tuples.
    Each page becomes its own Document so page numbers are preserved.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError("PyMuPDF is required: pip install pymupdf") from exc

    pages: List[Tuple[str, dict]] = []
    with fitz.open(str(path)) as doc:
        author = doc.metadata.get("author", "")
        title = doc.metadata.get("title", "") or path.stem
        total_pages = len(doc)
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            cleaned = clean_text(text)
            if not cleaned:
                continue  # skip blank/image-only pages
            meta = {
                "source": path.name,
                "file_path": str(path),
                "page": page_num,
                "total_pages": total_pages,
                "title": title,
                "author": author,
                "file_type": "pdf",
            }
            pages.append((cleaned, meta))
    logger.info("PDF '%s' → %d non-empty pages extracted.", path.name, len(pages))
    return pages


def _load_docx(path: Path) -> List[Tuple[str, dict]]:
    """
    Parse a DOCX file using python-docx.

    DOCX files don't have natural page boundaries, so we treat
    the whole document as a single (text, metadata) pair.
    """
    try:
        import docx
    except ImportError as exc:
        raise ImportError("python-docx is required: pip install python-docx") from exc

    doc = docx.Document(str(path))

    # Extract paragraph text, preserving paragraph breaks
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    full_text = "\n\n".join(paragraphs)
    cleaned = clean_text(full_text)

    # Core properties (may not exist in all DOCX files)
    try:
        props = doc.core_properties
        author = props.author or ""
        title = props.title or path.stem
    except Exception:
        author, title = "", path.stem

    meta = {
        "source": path.name,
        "file_path": str(path),
        "page": 1,
        "total_pages": 1,
        "title": title,
        "author": author,
        "file_type": "docx",
    }
    logger.info("DOCX '%s' → %d characters extracted.", path.name, len(cleaned))
    return [(cleaned, meta)]


def _load_txt(path: Path) -> List[Tuple[str, dict]]:
    """
    Load a plain-text file, trying UTF-8 then falling back to latin-1.
    """
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    text = ""
    for enc in encodings:
        try:
            text = path.read_text(encoding=enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue

    cleaned = clean_text(text)
    meta = {
        "source": path.name,
        "file_path": str(path),
        "page": 1,
        "total_pages": 1,
        "title": path.stem,
        "author": "",
        "file_type": "txt",
    }
    logger.info("TXT '%s' → %d characters extracted.", path.name, len(cleaned))
    return [(cleaned, meta)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

LOADER_MAP = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".doc": _load_docx,
    ".txt": _load_txt,
    ".md": _load_txt,
}


def load_document(file_path: str | Path) -> List[Document]:
    """
    Load a single document and return a list of LangChain Document objects.

    Each returned Document has:
        - page_content: cleaned text
        - metadata:     dict with source, page, title, author, file_type, …

    Args:
        file_path: Absolute or relative path to the file.

    Returns:
        List of Document objects (one per page for PDF, one for DOCX/TXT).

    Raises:
        ValueError: if the file extension is not supported.
        FileNotFoundError: if the file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in LOADER_MAP:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {list(LOADER_MAP.keys())}"
        )

    file_meta = extract_file_metadata(path)
    loader_fn = LOADER_MAP[ext]

    try:
        pages = loader_fn(path)
    except Exception as exc:
        logger.error("Failed to load '%s': %s", path.name, exc, exc_info=True)
        raise

    documents: List[Document] = []
    for text, meta in pages:
        # Merge file-level metadata into page-level metadata
        merged_meta = {**file_meta, **meta}
        documents.append(Document(page_content=text, metadata=merged_meta))

    return documents


def load_documents_from_dir(directory: str | Path) -> List[Document]:
    """
    Load all supported documents from a directory (non-recursive).

    Args:
        directory: Path to folder containing documents.

    Returns:
        Flattened list of Document objects from all files.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    all_docs: List[Document] = []
    for file in sorted(dir_path.iterdir()):
        if file.suffix.lower() in LOADER_MAP:
            try:
                docs = load_document(file)
                all_docs.extend(docs)
                logger.info("Loaded %d documents from '%s'.", len(docs), file.name)
            except Exception as exc:
                logger.warning("Skipping '%s' due to error: %s", file.name, exc)

    logger.info("Total documents loaded from directory: %d", len(all_docs))
    return all_docs


def load_from_bytes(file_bytes: bytes, filename: str, save_dir: str | Path = "data/documents") -> List[Document]:
    """
    Accept raw file bytes (e.g. from Streamlit uploader), save to disk, then load.

    Args:
        file_bytes: Raw bytes from the uploaded file.
        filename:   Original filename (used to detect extension).
        save_dir:   Directory where the file will be persisted.

    Returns:
        List of Document objects.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    dest = save_path / filename

    with open(dest, "wb") as f:
        f.write(file_bytes)

    logger.info("Saved uploaded file to '%s' (%d bytes).", dest, len(file_bytes))
    return load_document(dest)
