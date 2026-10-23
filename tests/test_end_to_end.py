"""End-to-end pipeline tests (Requirements 1 and 12)."""

import pytest
from unittest.mock import MagicMock
from src.answer_generator import AnswerGenerator, RAGAnswerModel
from src.citation_validator import CitationValidator
from src.chunking import chunk_pages
from src.retriever import ChromaVectorStoreManager
from src.embeddings import DeterministicMockEmbeddings
from config import PDF_FILENAME


def test_end_to_end_supported_question_flow(tmp_path):
    """Requirements 1 and 12: Ingestion -> Retrieval -> Generation -> Citation Validation -> Final Answer."""
    store_dir = tmp_path / "e2e_chroma"
    mock_embed = DeterministicMockEmbeddings()
    retriever_mgr = ChromaVectorStoreManager(
        persist_dir=store_dir,
        collection_name="e2e_col",
        embedding_function=mock_embed,
    )

    # 1. Document ingestion
    raw_pages = [
        {
            "page_index": 9,
            "page_number": 10,
            "source": PDF_FILENAME,
            "text": (
                "Typical baseline inspection intervals (6B.03/7E.03): "
                "Hot gas path inspection 24,000 hrs or 1200 starts. "
                "Major inspection 48,000 hrs or 2400 starts."
            ),
        }
    ]
    chunks = chunk_pages(raw_pages)
    retriever_mgr.index_chunks(chunks)
    assert retriever_mgr.count() > 0

    # 2. Retrieval
    query = "What is the baseline hot gas path inspection interval for 7E.03?"
    retrieved = retriever_mgr.retrieve(query, top_k=1)
    assert len(retrieved) == 1
    assert retrieved[0]["page_number"] == 10

    # 3. Mocked deterministic generator response
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = RAGAnswerModel(
        is_supported=True,
        answer="The baseline hot gas path inspection interval for 7E.03 is 24,000 hours or 1,200 starts, whichever occurs first.",
        cited_pages=[10],
        supporting_quotes=["Hot gas path inspection 24,000 hrs or 1200 starts"],
    )
    mock_llm.with_structured_output.return_value = mock_structured

    generator = AnswerGenerator(
        llm=mock_llm,
        citation_validator=CitationValidator(max_pdf_pages=60, source_filename=PDF_FILENAME),
    )

    # 4. End-to-end execution
    result = generator.generate_answer(query, retrieved)

    # 5. Assertions
    assert result["is_grounded"] is True
    assert "24,000 hours" in result["answer"]
    assert result["citation"] == f"Source: {PDF_FILENAME}, PDF page 10"
    assert result["cited_pages"] == [10]
