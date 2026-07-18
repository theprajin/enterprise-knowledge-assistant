"""
AI query metrics tracking for observability.

Tracks per-query metrics (latency, tokens, cost) and provides
aggregation methods for dashboards and monitoring.
"""

import logging
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# ── Pricing Table (USD per 1M tokens) ────────────────────────────────────
# Source: https://ai.google.dev/pricing
MODEL_PRICING = {
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
}


@dataclass
class QueryMetrics:
    """Metrics for a single RAG query execution."""

    query_id: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    model: str = ""
    prompt_name: str = ""
    prompt_version: int = 1

    # Timing
    latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0

    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Cost
    estimated_cost_usd: float = 0.0

    # Retrieval
    retrieval_count: int = 0
    conversation_id: Optional[str] = None

    # Guardrails
    guardrails_passed: bool = True
    guardrail_violations: int = 0


def estimate_cost(
    model: str, input_tokens: int, output_tokens: int
) -> float:
    """
    Estimate the cost of a query based on model pricing.
    
    Returns:
        Estimated cost in USD.
    """
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return 0.0

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 8)


class MetricsTracker:
    """
    In-memory metrics tracker with aggregation.
    
    Stores the last N queries (default 1000) in a ring buffer
    and provides summary statistics for monitoring.
    """

    def __init__(self, max_entries: int = 1000):
        self._metrics: deque[QueryMetrics] = deque(maxlen=max_entries)
        self._total_queries: int = 0
        self._total_tokens: int = 0
        self._total_cost: float = 0.0
        self._total_errors: int = 0

    def record(self, metrics: QueryMetrics) -> None:
        """Record a query's metrics."""
        self._metrics.append(metrics)
        self._total_queries += 1
        self._total_tokens += metrics.total_tokens
        self._total_cost += metrics.estimated_cost_usd

        logger.info(
            f"Metrics recorded: query_id={metrics.query_id} "
            f"latency={metrics.latency_ms}ms "
            f"tokens={metrics.total_tokens} "
            f"cost=${metrics.estimated_cost_usd:.6f}"
        )

    def record_error(self) -> None:
        """Record a query error for error rate calculation."""
        self._total_errors += 1

    def get_summary(self, last_n: int = 100) -> dict:
        """
        Get aggregated metrics summary.
        
        Args:
            last_n: Number of recent queries to aggregate over.
        """
        recent = list(self._metrics)[-last_n:]

        if not recent:
            return {
                "total_queries": self._total_queries,
                "period_queries": 0,
                "avg_latency_ms": 0,
                "p95_latency_ms": 0,
                "total_tokens_used": self._total_tokens,
                "total_estimated_cost_usd": round(self._total_cost, 6),
                "avg_tokens_per_query": 0,
                "error_rate": 0,
            }

        latencies = sorted([m.latency_ms for m in recent])
        tokens = [m.total_tokens for m in recent]
        p95_index = int(len(latencies) * 0.95)

        error_rate = 0.0
        if self._total_queries > 0:
            error_rate = round(self._total_errors / self._total_queries, 4)

        return {
            "total_queries": self._total_queries,
            "period_queries": len(recent),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
            "p95_latency_ms": round(latencies[min(p95_index, len(latencies) - 1)], 2),
            "total_tokens_used": self._total_tokens,
            "total_estimated_cost_usd": round(self._total_cost, 6),
            "avg_tokens_per_query": round(sum(tokens) / len(tokens), 1),
            "error_rate": error_rate,
        }

    def get_recent(self, n: int = 20) -> list[dict]:
        """Get the last N query metrics as dicts."""
        recent = list(self._metrics)[-n:]
        return [asdict(m) for m in reversed(recent)]

    def get_health(self) -> dict:
        """Get AI system health status."""
        summary = self.get_summary(last_n=50)
        
        # Simple health determination
        status = "healthy"
        if summary["error_rate"] > 0.1:
            status = "degraded"
        if summary["error_rate"] > 0.5:
            status = "unhealthy"
        if summary["avg_latency_ms"] > 10000:
            status = "slow"

        return {
            "status": status,
            "avg_latency_ms": summary["avg_latency_ms"],
            "p95_latency_ms": summary["p95_latency_ms"],
            "error_rate": summary["error_rate"],
            "total_queries": summary["total_queries"],
        }


@lru_cache()
def get_metrics_tracker() -> MetricsTracker:
    """Get the singleton metrics tracker instance."""
    return MetricsTracker()
