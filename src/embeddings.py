"""Embeddings factory module supporting FastEmbed (local), Google GenAI, and Mock embeddings.

Enables seamless switching between local high-speed embeddings, live cloud embeddings,
and deterministic mock embeddings for offline testing.
"""

import hashlib
import logging
import os
from typing import List, Optional
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class FastEmbeddingsWrapper(Embeddings):
    """Fast, local ONNX-based text embeddings with zero rate limits or API fees."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = list(self.model.embed(texts))
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        embedding = list(self.model.embed([text]))[0]
        return embedding.tolist()


class DeterministicMockEmbeddings(Embeddings):
    """Deterministic, lightweight pseudo-embeddings for offline testing and verification.
    
    Produces a 64-dimensional float vector derived deterministically from the SHA256
    hash of the input text. Requires zero network access and runs instantaneously.
    """

    def __init__(self, dimension: int = 64):
        self.dimension = dimension

    def _hash_to_vector(self, text: str) -> List[float]:
        hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
        vec = []
        for i in range(self.dimension):
            byte_val = hash_bytes[i % len(hash_bytes)]
            norm_val = (byte_val / 127.5) - 1.0
            vec.append(round(norm_val, 4))
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_to_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._hash_to_vector(text)


def get_embedding_function(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Embeddings:
    """Return configured embedding instance.

    Supported providers:
        - 'fastembed' or 'local': Fast, local ONNX-based BAAI/bge-small-en-v1.5. No rate limits.
        - 'gemini': Google Generative AI embeddings via Gemini API.
        - 'mock': Deterministic hash vectors for offline unit testing.
    """
    selected_provider = provider or os.getenv("EMBEDDING_PROVIDER", "fastembed").lower()

    if selected_provider == "mock":
        logger.info("Using DeterministicMockEmbeddings for testing.")
        return DeterministicMockEmbeddings()

    if selected_provider in ("fastembed", "local"):
        logger.info("Using FastEmbeddingsWrapper (local ONNX, zero rate limits).")
        return FastEmbeddingsWrapper()

    if selected_provider == "gemini":
        from config import GEMINI_API_KEY, EMBEDDING_MODEL
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        key = api_key or GEMINI_API_KEY
        model = model_name or EMBEDDING_MODEL
        if not key:
            logger.warning("No Gemini API key found; falling back to local FastEmbed.")
            return FastEmbeddingsWrapper()
        logger.info(f"Using Google Generative AI Embeddings: {model}")
        return GoogleGenerativeAIEmbeddings(model=model, google_api_key=key)

    # Default fallback
    return FastEmbeddingsWrapper()
