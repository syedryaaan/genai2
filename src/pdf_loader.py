"""PDF text extraction and page metadata preservation using PyMuPDF.

Extracts text page-by-page from GER-3620-P.pdf while preserving original page indices,
page numbers, and document metadata. Does not discard extraction errors.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List
import pymupdf  # PyMuPDF

logger = logging.getLogger(__name__)


class PDFLoadError(Exception):
    """Exception raised when loading or extracting a PDF fails."""
    pass


def get_pdf_metadata(pdf_path: str | Path) -> Dict[str, Any]:
    """Retrieve document metadata, file size, and page count."""
    path = Path(pdf_path)
    if not path.exists():
        raise PDFLoadError(f"PDF file does not exist: {path}")
    if path.stat().st_size == 0:
        raise PDFLoadError(f"PDF file is empty: {path}")

    try:
        doc = pymupdf.open(str(path))
        meta = {
            "filename": path.name,
            "filepath": str(path.resolve()),
            "file_size_bytes": path.stat().st_size,
            "page_count": len(doc),
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "creator": doc.metadata.get("creator", ""),
            "producer": doc.metadata.get("producer", ""),
            "creation_date": doc.metadata.get("creationDate", ""),
            "format": doc.metadata.get("format", ""),
        }
        doc.close()
        return meta
    except Exception as e:
        raise PDFLoadError(f"Failed to inspect PDF metadata: {e}") from e


def load_pdf_pages(pdf_path: str | Path, min_char_threshold: int = 10) -> List[Dict[str, Any]]:
    """Extract page-level text and metadata from the target PDF.

    Args:
        pdf_path: Path to the target PDF file.
        min_char_threshold: Flag warning if page has fewer characters than this.

    Returns:
        List of dictionaries with keys:
            - page_index: int (0-based)
            - page_number: int (1-based, verified PDF page number)
            - text: str (raw extracted text)
            - char_count: int
            - source: str (filename)
            - is_suspiciously_short: bool
    """
    path = Path(pdf_path)
    if not path.exists():
        raise PDFLoadError(f"PDF file not found at: {path}")

    try:
        doc = pymupdf.open(str(path))
    except Exception as e:
        raise PDFLoadError(f"Could not open PDF file '{path}': {e}") from e

    total_pages = len(doc)
    if total_pages == 0:
        doc.close()
        raise PDFLoadError(f"PDF file '{path}' contains 0 pages.")

    pages_data: List[Dict[str, Any]] = []

    for page_idx in range(total_pages):
        page = doc[page_idx]
        text = page.get_text("text")
        cleaned_text = text.strip()
        char_count = len(cleaned_text)

        is_short = char_count < min_char_threshold
        if is_short:
            logger.warning(
                f"Page {page_idx + 1} has unusually short text ({char_count} chars). "
                f"Check if it is a blank or graphic-only page."
            )

        pages_data.append({
            "page_index": page_idx,
            "page_number": page_idx + 1,  # 1-based human-readable PDF page
            "text": cleaned_text,
            "char_count": char_count,
            "source": path.name,
            "is_suspiciously_short": is_short,
        })

    doc.close()
    logger.info(f"Successfully extracted {len(pages_data)} pages from '{path.name}'.")
    return pages_data
