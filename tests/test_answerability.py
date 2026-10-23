"""Unit tests for answerability, strict grounding, and abstention (Requirements 2, 3, 4, 8, 9)."""

import pytest
from unittest.mock import MagicMock
from src.answer_generator import AnswerGenerator, RAGAnswerModel
from src.citation_validator import CitationValidator
from config import NOT_FOUND_RESPONSE, PDF_FILENAME


@pytest.fixture
def mock_citation_validator():
    return CitationValidator(max_pdf_pages=60, source_filename=PDF_FILENAME)


def test_invalid_input_blank_query(mock_citation_validator):
    """Requirement 9: Blank questions and malformed requests return exactly 'Not found'."""
    gen = AnswerGenerator(citation_validator=mock_citation_validator)
    
    assert gen.generate_answer("", [])["answer"] == NOT_FOUND_RESPONSE
    assert gen.generate_answer("   ", [])["answer"] == NOT_FOUND_RESPONSE
    assert gen.generate_answer("\n\t", [])["answer"] == NOT_FOUND_RESPONSE


def test_absent_answer_empty_evidence(mock_citation_validator):
    """Requirement 2: Question where no evidence exists returns exactly 'Not found'."""
    gen = AnswerGenerator(citation_validator=mock_citation_validator)
    res = gen.generate_answer("What is the battery capacity of the Mars Rover?", [])
    assert res["answer"] == NOT_FOUND_RESPONSE
    assert res["is_grounded"] is False
    assert res["citation"] == ""


def test_misleading_similarity_rejection(mock_citation_validator):
    """Requirement 3: Retrieved passages share some keywords (rotor, turbine) but do not prove requested fact."""
    # LLM returns unsupported because evidence doesn't answer the specific question
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = RAGAnswerModel(
        is_supported=False,
        answer="Not found",
        cited_pages=[],
        supporting_quotes=[],
    )
    mock_llm.with_structured_output.return_value = mock_structured

    gen = AnswerGenerator(llm=mock_llm, citation_validator=mock_citation_validator)

    chunks = [{
        "chunk_id": "c1",
        "page_number": 14,
        "text": "Rotors that are cold when startup commences experience transient thermal stresses as the turbine is brought on line.",
    }]

    # Question uses turbine/rotor terms but asks for something completely absent (quantum cooling)
    res = gen.generate_answer("What is the quantum cryogenic cooling temperature for the turbine rotor?", chunks)
    assert res["answer"] == NOT_FOUND_RESPONSE
    assert res["is_grounded"] is False


def test_partial_evidence_abstention(mock_citation_validator):
    """Requirement 4: Multi-part question where only part is in PDF -> system abstains."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    # LLM correctly abstains because東京 gold price is not in PDF
    mock_structured.invoke.return_value = RAGAnswerModel(
        is_supported=False,
        answer="Not found",
        cited_pages=[],
        supporting_quotes=[],
    )
    mock_llm.with_structured_output.return_value = mock_structured

    gen = AnswerGenerator(llm=mock_llm, citation_validator=mock_citation_validator)

    chunks = [{
        "chunk_id": "c1",
        "page_number": 10,
        "text": "Typical baseline inspection intervals (6B.03/7E.03): Hot gas path inspection 24,000 hrs or 1200 starts.",
    }]

    res = gen.generate_answer(
        "What is the baseline HGP interval for 7E.03 and what is the current price of gold in Tokyo?",
        chunks,
    )
    assert res["answer"] == NOT_FOUND_RESPONSE
    assert res["is_grounded"] is False


def test_ambiguous_or_contradictory_abstention(mock_citation_validator):
    """Requirement 8: Ambiguous question with insufficient specifics returns exactly 'Not found'."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = RAGAnswerModel(
        is_supported=False,
        answer="Not found",
        cited_pages=[],
        supporting_quotes=[],
    )
    mock_llm.with_structured_output.return_value = mock_structured

    gen = AnswerGenerator(llm=mock_llm, citation_validator=mock_citation_validator)
    chunks = [{
        "chunk_id": "c1",
        "page_number": 33,
        "text": "Intervals assume Base Tfire. Precedent technology may have shorter inspection intervals.",
    }]

    res = gen.generate_answer("When should I inspect the secret custom variant machine?", chunks)
    assert res["answer"] == NOT_FOUND_RESPONSE
