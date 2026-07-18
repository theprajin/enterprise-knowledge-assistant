"""
LLM output validators for response quality assurance.

Validates LLM responses for emptiness, length bounds, and common
hallucination patterns to ensure output quality before returning to users.
"""

import logging

logger = logging.getLogger(__name__)


def validate_response_not_empty(response: str) -> list[str]:
    """
    Check that the LLM response is not empty or whitespace-only.
    
    Returns:
        List of violation descriptions (empty if valid).
    """
    violations = []
    if not response or not response.strip():
        violations.append("Response is empty or whitespace-only")
        logger.warning("Guardrail violation: empty response detected")
    return violations


def validate_response_length(
    response: str,
    min_length: int = 10,
    max_length: int = 10000,
) -> list[str]:
    """
    Check that the response length is within acceptable bounds.
    
    Returns:
        List of violation descriptions (empty if valid).
    """
    violations = []
    length = len(response.strip())

    if length < min_length:
        violations.append(
            f"Response too short ({length} chars, minimum is {min_length})"
        )
        logger.warning(f"Guardrail violation: response too short ({length} chars)")

    if length > max_length:
        violations.append(
            f"Response too long ({length} chars, maximum is {max_length})"
        )
        logger.warning(f"Guardrail violation: response too long ({length} chars)")

    return violations


def validate_no_hallucination_markers(
    response: str,
    context_provided: bool = True,
) -> list[str]:
    """
    Flag common hallucination patterns where the LLM claims it cannot
    access information when context was actually provided.
    
    Returns:
        List of violation descriptions (empty if valid).
    """
    if not context_provided:
        return []

    violations = []
    response_lower = response.lower()

    # Patterns that suggest the LLM is ignoring provided context
    hallucination_markers = [
        "as an ai, i don't have access",
        "i don't have access to real-time",
        "i cannot browse the internet",
        "i don't have the ability to access",
        "my training data only goes up to",
        "i'm not able to access external",
        "i cannot access your documents",
        "i do not have access to your files",
    ]

    for marker in hallucination_markers:
        if marker in response_lower:
            violations.append(
                f"Possible hallucination: LLM claims lack of access despite context being provided"
            )
            logger.warning(
                f"Guardrail violation: hallucination marker detected — '{marker}'"
            )
            break  # One violation is enough

    return violations
