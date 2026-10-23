"""Unit tests for PDF loading and text extraction (Requirement 10)."""

import pytest
from pathlib import Path
from config import PDF_PATH, PDF_FILENAME
from src.pdf_loader import get_pdf_metadata, load_pdf_pages, PDFLoadError


def test_pdf_metadata_extraction():
    """Verify metadata extraction from the actual GER-3620-P.pdf."""
    assert PDF_PATH.exists(), f"PDF must exist at {PDF_PATH}"
    meta = get_pdf_metadata(PDF_PATH)
    assert meta["filename"] == PDF_FILENAME
    assert meta["page_count"] == 60
    assert meta["file_size_bytes"] > 5_000_000
    assert "GER-3620" in meta["title"]


def test_pdf_page_level_extraction():
    """Verify that pages are extracted with correct 1-based page numbers and content."""
    pages = load_pdf_pages(PDF_PATH)
    assert len(pages) == 60

    # Test page numbering
    for i, p in enumerate(pages):
        assert p["page_index"] == i
        assert p["page_number"] == i + 1
        assert p["source"] == PDF_FILENAME
        assert isinstance(p["text"], str)

    # Page 1 should contain title text
    assert "HEAVY-DUTY GAS" in pages[0]["text"]
    assert "TURBINE OPERATING" in pages[0]["text"]

    # Page 5 should contain Introduction text
    assert "Introduction" in pages[4]["text"]


def test_pdf_loader_invalid_file(tmp_path):
    """Test loader behavior with nonexistent file."""
    non_existent = tmp_path / "does_not_exist.pdf"
    with pytest.raises(PDFLoadError):
        load_pdf_pages(non_existent)
