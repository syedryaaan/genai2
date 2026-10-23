"""Unit tests for citation validation and formatting (Requirements 5 and 6)."""

import pytest
from src.citation_validator import CitationValidator
from config import PDF_FILENAME


@pytest.fixture
def validator():
    return CitationValidator(max_pdf_pages=60, source_filename=PDF_FILENAME)


@pytest.fixture
def sample_retrieved_chunks():
    return [
        {
            "chunk_id": "chunk_1",
            "page_number": 10,
            "text": "Typical baseline inspection intervals for 6B.03/7E.03 are 24,000 hours or 1200 starts for hot gas path.",
        },
        {
            "chunk_id": "chunk_2",
            "page_number": 12,
            "text": "Peak fire severity factor Ap = exp(0.018 * delta_Tf) for B/E-class gas turbines.",
        },
    ]


def test_valid_citation(validator, sample_retrieved_chunks):
    """Test verification of a legitimate cited page with matching evidence."""
    is_valid, pages, reason = validator.validate_citations(
        candidate_pages=[10],
        retrieved_chunks=sample_retrieved_chunks,
        answer_text="Baseline hot gas path interval is 24,000 hours or 1200 starts.",
    )
    assert is_valid is True
    assert pages == [10]
    formatted = validator.format_citation(pages)
    assert formatted == f"Source: {PDF_FILENAME}, PDF page 10"


def test_invalid_page_beyond_boundary(validator, sample_retrieved_chunks):
    """Requirement 5: Candidate citation points to nonexistent page (e.g. page 99)."""
    is_valid, pages, reason = validator.validate_citations(
        candidate_pages=[99],
        retrieved_chunks=sample_retrieved_chunks,
        answer_text="Some assertion.",
    )
    assert is_valid is False
    assert len(pages) == 0
    assert "does not exist in the PDF" in reason


def test_invalid_negative_or_zero_page(validator, sample_retrieved_chunks):
    """Requirement 5: Page 0 or negative page is rejected."""
    is_valid, pages, reason = validator.validate_citations(
        candidate_pages=[0],
        retrieved_chunks=sample_retrieved_chunks,
        answer_text="Some assertion.",
    )
    assert is_valid is False
    assert len(pages) == 0


def test_unsupported_citation_not_in_retrieved(validator, sample_retrieved_chunks):
    """Requirement 6: Page exists in PDF (page 25) but was never retrieved in evidence."""
    is_valid, pages, reason = validator.validate_citations(
        candidate_pages=[25],
        retrieved_chunks=sample_retrieved_chunks,
        answer_text="Baseline hot gas path interval.",
    )
    assert is_valid is False
    assert len(pages) == 0
    assert "was not in the retrieved passages" in reason


def test_unsupported_citation_unsubstantiated_text(validator, sample_retrieved_chunks):
    """Requirement 6: Page was retrieved (page 12), but content does not substantiate unrelated claim."""
    is_valid, pages, reason = validator.validate_citations(
        candidate_pages=[12],
        retrieved_chunks=sample_retrieved_chunks,
        answer_text="The submarine reactor coolant temperature operates at 450 kelvin.",
    )
    assert is_valid is False
    assert len(pages) == 0
    assert "does not substantiate" in reason


def test_multiple_citations_formatting(validator):
    """Test multi-page citation formatting."""
    formatted = validator.format_citation([10, 33])
    assert formatted == f"Source: {PDF_FILENAME}, PDF pages 10, 33"
