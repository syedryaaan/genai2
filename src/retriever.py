"""ChromaDB vector store management, ingestion, and document retrieval.

Manages persistent local storage using ChromaDB, ensures metadata preservation,
and retrieves top-k relevant chunks with score diagnostics.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import CHROMA_DIR, CHROMA_COLLECTION_NAME, TOP_K
from src.embeddings import get_embedding_function

logger = logging.getLogger(__name__)


class ChromaVectorStoreManager:
    """Manages the persistent ChromaDB collection and queries."""

    def __init__(
        self,
        persist_dir: str | Path = CHROMA_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
        embedding_function: Optional[Any] = None,
    ):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embedding_function = embedding_function or get_embedding_function()
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._vector_store: Optional[Chroma] = None

    @property
    def vector_store(self) -> Chroma:
        """Lazy-load the LangChain Chroma vector store."""
        if self._vector_store is None:
            self._vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
                persist_directory=str(self.persist_dir),
            )
        return self._vector_store

    def count(self) -> int:
        """Return total number of chunks currently stored in the collection."""
        try:
            return self.vector_store._collection.count()
        except Exception:
            return 0

    def index_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 100) -> int:
        """Store chunk documents and metadata in ChromaDB.

        Args:
            chunks: List of chunk dictionaries containing text, chunk_id, and metadata.
            batch_size: Ingestion batch size.

        Returns:
            Count of indexed chunks.
        """
        if not chunks:
            logger.warning("No chunks provided to index.")
            return 0

        documents = []
        ids = []

        for chunk in chunks:
            doc = Document(
                page_content=chunk["text"],
                metadata={
                    "chunk_id": chunk["chunk_id"],
                    "source": chunk["source"],
                    "page_index": chunk["page_index"],
                    "page_number": chunk["page_number"],
                    "section_heading": chunk.get("section_heading", "General"),
                },
            )
            documents.append(doc)
            ids.append(chunk["chunk_id"])

        logger.info(f"Adding {len(documents)} documents to ChromaDB collection '{self.collection_name}'...")
        
        # Batch insert to avoid rate limits or memory pressure
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            self.vector_store.add_documents(documents=batch_docs, ids=batch_ids)

        logger.info(f"Successfully indexed {len(documents)} documents. Total in store: {self.count()}")
        return len(documents)

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant chunks with similarity score.

        Returns:
            List of dictionaries:
                - chunk_id: str
                - text: str
                - source: str
                - page_number: int (1-based)
                - page_index: int (0-based)
                - section_heading: str
                - score: float (relevance distance)
        """
        if not query or not query.strip():
            logger.warning("Empty query passed to retriever.")
            return []

        try:
            results_with_scores = self.vector_store.similarity_search_with_relevance_scores(
                query.strip(),
                k=top_k,
            )
        except Exception as e:
            # Fall back to standard similarity search if relevance score isn't supported by distance metric
            logger.debug(f"Relevance score query fallback: {e}")
            results = self.vector_store.similarity_search(query.strip(), k=top_k)
            results_with_scores = [(doc, 1.0) for doc in results]

        retrieved_chunks: List[Dict[str, Any]] = []
        for doc, score in results_with_scores:
            retrieved_chunks.append({
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "text": doc.page_content,
                "source": doc.metadata.get("source", "GER-3620-P.pdf"),
                "page_number": int(doc.metadata.get("page_number", 1)),
                "page_index": int(doc.metadata.get("page_index", 0)),
                "section_heading": doc.metadata.get("section_heading", "General"),
                "score": float(score) if score is not None else 1.0,
            })

        return retrieved_chunks

    def reset_collection(self) -> None:
        """Clear all entries in the collection for a fresh rebuild."""
        try:
            self.vector_store._client.delete_collection(self.collection_name)
            self._vector_store = None
            logger.info(f"Reset collection '{self.collection_name}'.")
        except Exception as e:
            logger.warning(f"Error resetting collection (may not exist yet): {e}")
            self._vector_store = None
