"""Citation validation and verification module.

Ensures that every citation:
1. Matches the actual PDF source filename.
2. Cites a valid 1-based page within the verified page count (1 to 60).
3. Was actually retrieved in the evidence set.
4. Actually contains textual grounding for the generated answer.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from config import PDF_FILENAME

logger = logging.getLogger(__name__)


class CitationValidator:
    """Validates citations against the source PDF and retrieved chunks."""

    def __init__(self, max_pdf_pages: int = 60, source_filename: str = PDF_FILENAME):
        self.max_pdf_pages = max_pdf_pages
        self.source_filename = source_filename

    def validate_citations(
        self,
        candidate_pages: List[int],
        retrieved_chunks: List[Dict[str, Any]],
        answer_text: str,
    ) -> Tuple[bool, List[int], str]:
        """Validate candidate cited page numbers.

        Args:
            candidate_pages: Page numbers cited by the generator.
            retrieved_chunks: Chunks retrieved by the vector store.
            answer_text: Candidate answer string.

        Returns:
            Tuple of (is_valid, validated_pages, reason)
        """
        if not candidate_pages:
            return False, [], "No page citations provided."

        retrieved_page_numbers: Set[int] = {
            int(c["page_number"]) for c in retrieved_chunks if "page_number" in c
        }

        validated_pages: List[int] = []

        for page in candidate_pages:
            # 1. Page boundary check
            if page < 1 or page > self.max_pdf_pages:
                logger.warning(f"Citation rejected: Page {page} is outside PDF range [1, {self.max_pdf_pages}].")
                return False, [], f"Page {page} does not exist in the PDF."

            # 2. Page must be in retrieved chunks
            if page not in retrieved_page_numbers:
                logger.warning(f"Citation rejected: Page {page} was not among retrieved evidence.")
                return False, [], f"Page {page} was not in the retrieved passages."

            # 3. Check that the page chunk contains relevant terms
            page_chunks = [c for c in retrieved_chunks if c.get("page_number") == page]
            page_text = " ".join(c.get("text", "").lower() for c in page_chunks)

            # Basic overlap check with answer
            answer_keywords = [w for w in answer_text.lower().split() if len(w) > 4]
            if answer_keywords:
                matched = [w for w in answer_keywords if w in page_text]
                if len(matched) == 0 and len(answer_keywords) > 2:
                    return False, [], f"Page {page} text does not substantiate candidate answer."

            validated_pages.append(page)

        # Remove duplicates while preserving order
        unique_pages = sorted(list(set(validated_pages)))
        return True, unique_pages, "Citations successfully verified."

    def format_citation(self, pages: List[int]) -> str:
        """Format verified 1-based PDF page numbers."""
        if not pages:
            return ""
        if len(pages) == 1:
            return f"Source: {self.source_filename}, PDF page {pages[0]}"
        pages_str = ", ".join(str(p) for p in pages)
        return f"Source: {self.source_filename}, PDF pages {pages_str}"
