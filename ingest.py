"""Reproducible PDF Ingestion Pipeline for GER-3620-P.pdf into ChromaDB.

Usage:
    python ingest.py [--force] [--mock]
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

from config import (
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME,
    DATA_DIR,
    PDF_FILENAME,
    PDF_PATH,
    PDF_URL,
)
from src.chunking import chunk_pages
from src.embeddings import get_embedding_function
from src.pdf_loader import get_pdf_metadata, load_pdf_pages
from src.retriever import ChromaVectorStoreManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HASH_FILE = DATA_DIR / "pdf_hash.json"


def compute_file_sha256(filepath: Path) -> str:
    """Compute sha256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_ingestion(force: bool = False, use_mock: bool = False) -> bool:
    """Execute complete ingestion pipeline:
    1. Ensure PDF is present.
    2. Check hash to avoid duplicate ingestion.
    3. Extract pages.
    4. Chunk text with page isolation.
    5. Generate embeddings and persist in ChromaDB.
    6. Verify with test query.
    """
    logger.info("Starting ingestion workflow for GER-3620-P.pdf...")

    if not PDF_PATH.exists():
        logger.info(f"Downloading PDF from {PDF_URL} to {PDF_PATH}...")
        import urllib.request
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(PDF_URL, str(PDF_PATH))

    # Inspect metadata
    meta = get_pdf_metadata(PDF_PATH)
    logger.info(f"PDF verified: {meta['filename']} ({meta['page_count']} pages, {meta['file_size_bytes']} bytes)")

    current_hash = compute_file_sha256(PDF_PATH)

    # Initialize store manager
    from config import EMBEDDING_PROVIDER
    provider = "mock" if use_mock else EMBEDDING_PROVIDER
    embed_fn = get_embedding_function(provider=provider)
    manager = ChromaVectorStoreManager(
        persist_dir=CHROMA_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embed_fn,
    )

    current_count = manager.count()

    # Check for unchanged document
    if not force and HASH_FILE.exists() and current_count > 0:
        try:
            with open(HASH_FILE, "r") as f:
                cached_data = json.load(f)
            if cached_data.get("sha256") == current_hash and cached_data.get("provider") == provider:
                logger.info(
                    f"Index is up-to-date ({current_count} chunks indexed). Skipping ingestion. Use --force to rebuild."
                )
                return True
        except Exception as e:
            logger.warning(f"Error reading cache file: {e}")

    # Rebuilding index
    if force or current_count > 0:
        logger.info("Clearing previous vector store collection...")
        manager.reset_collection()

    # Load and extract pages
    pages = load_pdf_pages(PDF_PATH)
    logger.info(f"Extracted {len(pages)} pages.")

    # Chunk text
    chunks = chunk_pages(pages)
    logger.info(f"Generated {len(chunks)} text chunks.")

    # Ingest into ChromaDB
    indexed_count = manager.index_chunks(chunks)
    logger.info(f"Successfully stored {indexed_count} chunks into ChromaDB.")

    # Save hash and index metadata
    cache_record = {
        "filename": PDF_FILENAME,
        "sha256": current_hash,
        "page_count": meta["page_count"],
        "chunk_count": indexed_count,
        "provider": provider,
    }
    with open(HASH_FILE, "w") as f:
        json.dump(cache_record, f, indent=2)

    # Test retrieval
    test_query = "What are the baseline inspection intervals for 7E.03?"
    results = manager.retrieve(test_query, top_k=2)
    logger.info(f"Verification query ('{test_query}') retrieved {len(results)} chunks:")
    for r in results:
        logger.info(f"  - Chunk ID: {r['chunk_id']} (PDF Page {r['page_number']})")

    logger.info("Ingestion complete and verified successfully.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest GER-3620-P.pdf into ChromaDB")
    parser.add_argument("--force", action="store_true", help="Force rebuild of the vector index")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock embeddings for offline testing")
    args = parser.parse_args()

    success = run_ingestion(force=args.force, use_mock=args.mock)
    sys.exit(0 if success else 1)
