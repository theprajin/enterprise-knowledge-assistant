"""
LangChain callback handler for automatic token usage tracking.

Integrates with the MetricsTracker to capture token usage from
every LLM call without requiring manual instrumentation.
"""

import logging
from typing import Any
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback that captures token usage from LLM responses.
    
    Designed to be used as a per-request callback (not shared across requests)
    to capture token metrics for individual queries.
    """

    def __init__(self):
        super().__init__()
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.total_tokens: int = 0
        self.model_name: str = ""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Extract token usage from the LLM response."""
        if response.llm_output:
            usage = response.llm_output.get("usage_metadata") or {}

            # Google Gemini format
            self.input_tokens = usage.get("input_tokens", 0) or usage.get(
                "prompt_token_count", 0
            )
            self.output_tokens = usage.get("output_tokens", 0) or usage.get(
                "candidates_token_count", 0
            )
            self.total_tokens = usage.get("total_tokens", 0) or (
                self.input_tokens + self.output_tokens
            )

            # Try to get model name
            self.model_name = response.llm_output.get("model_name", "")

            logger.debug(
                f"Token usage captured: input={self.input_tokens}, "
                f"output={self.output_tokens}, total={self.total_tokens}"
            )

        # Also check generation-level token usage (some providers put it here)
        if not self.total_tokens and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    info = getattr(gen, "generation_info", {}) or {}
                    usage = info.get("usage_metadata", {})
                    if usage:
                        self.input_tokens = usage.get("input_tokens", 0)
                        self.output_tokens = usage.get("output_tokens", 0)
                        self.total_tokens = usage.get("total_tokens", 0)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Log LLM errors."""
        logger.error(f"LLM call failed: {error}")

    def get_usage(self) -> dict:
        """Get the captured token usage as a dict."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "model_name": self.model_name,
        }
