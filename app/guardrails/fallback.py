"""
Guardrail orchestrator and fallback logic.

Runs all validators and content filters on LLM responses, and provides
fallback responses when guardrails are violated.
"""

import logging
from dataclasses import dataclass, field

from app.guardrails.validators import (
    validate_response_not_empty,
    validate_response_length,
    validate_no_hallucination_markers,
)
from app.guardrails.content_filter import filter_pii, filter_harmful_content

logger = logging.getLogger(__name__)

# Default fallback message when guardrails block the response
FALLBACK_MESSAGE = (
    "I was unable to generate a reliable answer for your question. "
    "This may be due to insufficient context in the available documents "
    "or content safety filters. Please try rephrasing your question or "
    "uploading additional relevant documents."
)


@dataclass
class GuardrailResult:
    """Result of running all guardrails on an LLM response."""

    passed: bool
    filtered_response: str
    violations: list[str] = field(default_factory=list)
    pii_detected: list[str] = field(default_factory=list)
    used_fallback: bool = False

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "violations": self.violations,
            "pii_detected": self.pii_detected,
            "used_fallback": self.used_fallback,
        }


def apply_guardrails(
    response: str,
    context_provided: bool = True,
    min_length: int = 10,
    max_length: int = 10000,
) -> GuardrailResult:
    """
    Run all guardrail checks on an LLM response.
    
    Pipeline:
    1. Validate non-empty
    2. Validate length bounds
    3. Check for hallucination markers
    4. Filter PII (mask, don't block)
    5. Check for harmful content (block if found)
    
    Args:
        response: The raw LLM response text.
        context_provided: Whether document context was provided to the LLM.
        min_length: Minimum acceptable response length.
        max_length: Maximum acceptable response length.
        
    Returns:
        GuardrailResult with the filtered response and violation details.
    """
    all_violations = []
    filtered_response = response

    # 1. Validate non-empty
    violations = validate_response_not_empty(response)
    all_violations.extend(violations)

    # If response is empty, return fallback immediately
    if violations:
        logger.warning("Guardrails: response was empty, using fallback")
        return GuardrailResult(
            passed=False,
            filtered_response=FALLBACK_MESSAGE,
            violations=all_violations,
            used_fallback=True,
        )

    # 2. Validate length
    violations = validate_response_length(response, min_length, max_length)
    all_violations.extend(violations)

    # 3. Check for hallucination markers
    violations = validate_no_hallucination_markers(response, context_provided)
    all_violations.extend(violations)

    # 4. Filter PII (mask but don't block)
    filtered_response, pii_types = filter_pii(filtered_response)

    # 5. Check for harmful content
    is_safe, harm_violations = filter_harmful_content(filtered_response)
    all_violations.extend(harm_violations)

    # Determine if we need to use fallback
    # Block only for: harmful content or empty response
    # Warn but pass for: length violations, hallucination markers, PII
    use_fallback = not is_safe

    if use_fallback:
        logger.warning(
            f"Guardrails: blocking response due to violations: {harm_violations}"
        )
        return GuardrailResult(
            passed=False,
            filtered_response=FALLBACK_MESSAGE,
            violations=all_violations,
            pii_detected=pii_types,
            used_fallback=True,
        )

    passed = len(all_violations) == 0

    if not passed:
        logger.info(
            f"Guardrails: response passed with warnings: {all_violations}"
        )

    return GuardrailResult(
        passed=passed,
        filtered_response=filtered_response,
        violations=all_violations,
        pii_detected=pii_types,
        used_fallback=False,
    )
