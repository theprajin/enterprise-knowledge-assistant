"""
API endpoints for AI observability and monitoring.

Exposes metrics aggregation, recent query details, and system health
for production monitoring dashboards.
"""

from fastapi import APIRouter, Query
from app.observability.tracker import get_metrics_tracker

router = APIRouter(
    prefix="/observability",
    tags=["observability"],
)


@router.get("/metrics")
async def get_metrics_summary(
    last_n: int = Query(100, ge=1, le=1000, description="Number of recent queries to aggregate")
):
    """
    Get aggregated AI system metrics.
    
    Returns average latency, P95 latency, total tokens, total cost,
    error rate, and query counts.
    """
    tracker = get_metrics_tracker()
    return tracker.get_summary(last_n=last_n)


@router.get("/metrics/recent")
async def get_recent_metrics(
    n: int = Query(20, ge=1, le=100, description="Number of recent query metrics to return")
):
    """Get detailed metrics for the N most recent queries."""
    tracker = get_metrics_tracker()
    return {
        "recent_queries": tracker.get_recent(n=n),
        "count": len(tracker.get_recent(n=n)),
    }


@router.get("/health")
async def get_ai_health():
    """
    Get AI system health status.
    
    Returns a health assessment based on error rate and latency thresholds.
    Statuses: healthy, degraded, unhealthy, slow.
    """
    tracker = get_metrics_tracker()
    return tracker.get_health()
