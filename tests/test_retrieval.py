"""Unit tests for ChromaDB vector store and retrieval (Requirements 7 and 11)."""

import pytest
from src.embeddings import DeterministicMockEmbeddings
from src.retriever import ChromaVectorStoreManager
from src.evidence_validator import EvidenceValidator


@pytest.fixture
def temp_vector_store(tmp_path):
    """Provides a fresh isolated Chroma vector store using mock embeddings."""
    store_dir = tmp_path / "test_chroma"
    mock_embed = DeterministicMockEmbeddings()
    manager = ChromaVectorStoreManager(
        persist_dir=store_dir,
        collection_name="test_collection",
        embedding_function=mock_embed,
    )
    return manager


def test_empty_retrieval_handling(temp_vector_store):
    """Requirement 7: Empty collection or empty query returns empty list, detected by EvidenceValidator."""
    # Blank query
    assert temp_vector_store.retrieve("") == []
    assert temp_vector_store.retrieve("   ") == []

    # Non-existent content in empty store
    results = temp_vector_store.retrieve("nonexistent topic")
    assert len(results) == 0
    assert EvidenceValidator.is_evidence_empty(results) is True


def test_metadata_preservation_and_retrieval(temp_vector_store):
    """Verify that chunks retain chunk_id, page_number, and source throughout retrieval."""
    test_chunks = [
        {
            "chunk_id": "doc1_p5_c0",
            "text": "Maintenance costs and machine availability are primary concerns for gas turbines.",
            "source": "GER-3620-P.pdf",
            "page_index": 4,
            "page_number": 5,
            "section_heading": "Introduction",
        },
        {
            "chunk_id": "doc1_p10_c0",
            "text": "Heavier hydrocarbon fuels have a maintenance factor ranging from three to four for residual fuels.",
            "source": "GER-3620-P.pdf",
            "page_index": 9,
            "page_number": 10,
            "section_heading": "Fuel",
        },
    ]

    temp_vector_store.index_chunks(test_chunks)
    assert temp_vector_store.count() == 2

    retrieved = temp_vector_store.retrieve("maintenance costs", top_k=2)
    assert len(retrieved) == 2
    for r in retrieved:
        assert "chunk_id" in r
        assert "page_number" in r
        assert "source" in r
        assert r["source"] == "GER-3620-P.pdf"


def test_vector_store_persistence(tmp_path):
    """Requirement 11: Vector store retains documents after closing and reopening instance."""
    store_dir = tmp_path / "persistent_chroma"
    mock_embed = DeterministicMockEmbeddings()

    # Step 1: Ingest into store
    mgr1 = ChromaVectorStoreManager(
        persist_dir=store_dir,
        collection_name="persist_col",
        embedding_function=mock_embed,
    )
    chunks = [
        {
            "chunk_id": "c1",
            "text": "Baseline hot gas path interval is 24,000 hours or 1200 starts.",
            "source": "GER-3620-P.pdf",
            "page_index": 5,
            "page_number": 6,
            "section_heading": "Service Factors",
        }
    ]
    mgr1.index_chunks(chunks)
    assert mgr1.count() == 1

    # Step 2: Reopen from same directory with a new manager instance
    mgr2 = ChromaVectorStoreManager(
        persist_dir=store_dir,
        collection_name="persist_col",
        embedding_function=mock_embed,
    )
    assert mgr2.count() == 1

    results = mgr2.retrieve("hot gas path interval", top_k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["page_number"] == 6
