"""Configuration settings for the PDF-based RAG system.
Loads environment variables and exposes paths, parameters, and model definitions.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

# PDF Specifications
PDF_FILENAME = "GER-3620-P.pdf"
PDF_PATH = DATA_DIR / PDF_FILENAME
PDF_URL = "https://www.gevernova.com/content/dam/gepower-new/global/en_US/downloads/gas-new-site/resources/reference/GER-3620-P.pdf"

# Text Chunking Settings
# Chunk size of 1000 characters (~200-250 words) preserves full technical
# inspection intervals, formula definitions, and table rows without truncation.
# Overlap of 200 characters prevents boundary splits across sentences.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ChromaDB Vector Store
CHROMA_COLLECTION_NAME = "ger_3620_rag"
TOP_K = int(os.getenv("TOP_K", "5"))

# LLM and Embedding Providers
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.8-flash")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "fastembed")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# Strict Grounding & Abstention Constants
NOT_FOUND_RESPONSE = "Not found"
