"""
Content filtering for LLM outputs.

Provides PII detection/masking and harmful content filtering
to ensure LLM responses are safe for enterprise use.
"""

import re
import logging

logger = logging.getLogger(__name__)

# ── PII Patterns ──────────────────────────────────────────────────────────

_PII_PATTERNS = {
    "email": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    ),
    "phone_us": re.compile(
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
    "ssn": re.compile(
        r"\b\d{3}-\d{2}-\d{4}\b"
    ),
    "credit_card": re.compile(
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
    ),
    "ip_address": re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    ),
}

_PII_MASKS = {
    "email": "[EMAIL_REDACTED]",
    "phone_us": "[PHONE_REDACTED]",
    "ssn": "[SSN_REDACTED]",
    "credit_card": "[CARD_REDACTED]",
    "ip_address": "[IP_REDACTED]",
}


def filter_pii(response: str) -> tuple[str, list[str]]:
    """
    Detect and mask PII patterns in the response.
    
    Args:
        response: The LLM response text.
        
    Returns:
        Tuple of (filtered_response, list_of_pii_types_found).
    """
    pii_found = []
    filtered = response

    for pii_type, pattern in _PII_PATTERNS.items():
        if pattern.search(filtered):
            pii_found.append(pii_type)
            filtered = pattern.sub(_PII_MASKS[pii_type], filtered)
            logger.info(f"PII filter: masked {pii_type} in LLM response")

    return filtered, pii_found


# ── Harmful Content Blocklist ─────────────────────────────────────────────

_HARMFUL_PATTERNS = [
    re.compile(r"\b(how\s+to\s+hack)\b", re.IGNORECASE),
    re.compile(r"\b(exploit\s+vulnerability)\b", re.IGNORECASE),
    re.compile(r"\b(bypass\s+security)\b", re.IGNORECASE),
    re.compile(r"\b(sql\s+injection\s+attack)\b", re.IGNORECASE),
    re.compile(r"\b(password\s+cracking)\b", re.IGNORECASE),
]


def filter_harmful_content(response: str) -> tuple[bool, list[str]]:
    """
    Check for harmful content patterns in the response.
    
    Args:
        response: The LLM response text.
        
    Returns:
        Tuple of (is_safe, list_of_violations).
    """
    violations = []

    for pattern in _HARMFUL_PATTERNS:
        match = pattern.search(response)
        if match:
            violations.append(
                f"Potentially harmful content detected: '{match.group()}'"
            )
            logger.warning(
                f"Content filter: harmful pattern detected — '{match.group()}'"
            )

    return len(violations) == 0, violations
