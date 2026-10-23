"""Text chunking module with page boundary preservation and stable chunk IDs.

Splits page-level documents into semantically coherent chunks using
RecursiveCharacterTextSplitter while strictly preserving 1-to-1 page boundaries.
No chunk combines text across separate PDF pages, guaranteeing exact citation traceability.
"""

import re
import logging
from typing import Any, Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def extract_section_header(text: str) -> str:
    """Heuristic to detect primary section heading from extracted page text."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    for line in lines[:5]:
        # Ignore running headers and page numbers
        if line.startswith("Doc Name") or line.startswith("Heavy-Duty Gas Turbine") or line.startswith("Non-Public"):
            continue
        if re.match(r"^\d+$", line):
            continue
        if len(line) < 60 and (line.isupper() or line.istitle() or line.startswith("Figure") or line.startswith("Appendix")):
            return line
    return "General"


def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """Split page documents into chunks while preserving exact page boundaries.

    Settings Rationale:
        - chunk_size = 1000 characters (~200-250 words): Fits discrete maintenance criteria,
          tables, and inspection rules without truncating equations or notes.
        - chunk_overlap = 200 characters: Preserves contextual flow across chunk boundaries
          within the same page.
        - Page-isolated chunking: Each page is split independently. Chunks never bridge
          two different pages, ensuring unambiguous page citations.

    Returns:
        List of chunk dictionaries:
            - chunk_id: str (stable unique ID, e.g. GER-3620-P.pdf_p5_c0)
            - text: str (chunk content)
            - source: str (PDF filename)
            - page_index: int (0-based)
            - page_number: int (1-based PDF page)
            - section_heading: str
            - char_count: int
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
        length_function=len,
    )

    all_chunks: List[Dict[str, Any]] = []

    for page in pages:
        page_text = page["text"]
        page_num = page["page_number"]
        page_idx = page["page_index"]
        source = page["source"]

        if not page_text:
            continue

        section_heading = extract_section_header(page_text)
        splits = text_splitter.split_text(page_text)

        for c_idx, split_text in enumerate(splits):
            chunk_id = f"{source}_p{page_num}_c{c_idx}"
            all_chunks.append({
                "chunk_id": chunk_id,
                "text": split_text,
                "source": source,
                "page_index": page_idx,
                "page_number": page_num,
                "section_heading": section_heading,
                "char_count": len(split_text),
            })

    logger.info(f"Generated {len(all_chunks)} chunks across {len(pages)} pages.")
    return all_chunks
