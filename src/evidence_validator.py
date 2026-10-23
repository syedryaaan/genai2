"""Evidence validation module for multi-tier grounding and sufficiency checking.

Implements layered validation:
1. Retrieval presence check (non-empty passages).
2. Query relevance & keyword containment signal.
3. Claim-to-passage alignment checking.
"""

import logging
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class EvidenceValidator:
    """Validates retrieved evidence against queries and generated claims."""

    @staticmethod
    def is_evidence_empty(retrieved_chunks: List[Dict[str, Any]]) -> bool:
        """Check if retrieved evidence is empty or contains only whitespace."""
        if not retrieved_chunks:
            return True
        total_text = "".join(chunk.get("text", "") for chunk in retrieved_chunks).strip()
        return len(total_text) == 0

    @staticmethod
    def check_retrieval_relevance(query: str, retrieved_chunks: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Perform heuristic pre-check on retrieved evidence."""
        if EvidenceValidator.is_evidence_empty(retrieved_chunks):
            return False, "No evidence retrieved."

        # Extract meaningful alphanumeric tokens (>3 chars) from query
        query_words = set(re.findall(r"\b[a-zA-Z0-9]{4,}\b", query.lower()))
        combined_text = " ".join(c.get("text", "").lower() for c in retrieved_chunks)

        if not query_words:
            return True, "Query contains no specific keywords."

        matches = [w for w in query_words if w in combined_text]
        match_ratio = len(matches) / len(query_words)

        # If zero keywords match across top retrieved chunks, evidence is completely disconnected
        if len(matches) == 0:
            return False, "Retrieved passages share zero key terms with query."

        return True, f"Keyword match ratio: {match_ratio:.2f}"

    @staticmethod
    def verify_claim_support(claim: str, retrieved_chunks: List[Dict[str, Any]]) -> bool:
        """Verify that a factual claim has textual grounding in at least one retrieved chunk."""
        if not claim or not claim.strip():
            return False

        claim_words = set(re.findall(r"\b[a-zA-Z0-9]{4,}\b", claim.lower()))
        if not claim_words:
            return True

        for chunk in retrieved_chunks:
            chunk_text = chunk.get("text", "").lower()
            matches = [w for w in claim_words if w in chunk_text]
            if len(matches) / len(claim_words) >= 0.5:
                return True

        return False
