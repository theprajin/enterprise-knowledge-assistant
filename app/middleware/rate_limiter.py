"""
In-memory sliding window rate limiter middleware.

Limits requests per IP address using a configurable requests-per-minute 
threshold. Returns 429 Too Many Requests with Retry-After header when 
the limit is exceeded.
"""

import time
import logging
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import settings

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Per-IP sliding window rate limiter.
    
    Tracks request timestamps per client IP and blocks requests
    that exceed the configured rate_limit_rpm (requests per minute).
    """

    def __init__(self, app, rpm: int | None = None):
        super().__init__(app)
        self.rpm = rpm or settings.rate_limit_rpm
        self.window_seconds = 60
        # {ip: [timestamp1, timestamp2, ...]}
        self._requests: dict[str, list[float]] = defaultdict(list)

    def _clean_old_requests(self, ip: str, now: float) -> None:
        """Remove request timestamps outside the current window."""
        cutoff = now - self.window_seconds
        self._requests[ip] = [
            ts for ts in self._requests[ip] if ts > cutoff
        ]

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health check endpoints
        if request.url.path in ("/health", "/", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        self._clean_old_requests(client_ip, now)

        if len(self._requests[client_ip]) >= self.rpm:
            # Calculate retry-after based on oldest request in window
            oldest = self._requests[client_ip][0]
            retry_after = int(self.window_seconds - (now - oldest)) + 1

            logger.warning(
                f"Rate limit exceeded for {client_ip}: "
                f"{len(self._requests[client_ip])}/{self.rpm} RPM"
            )

            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded. Please slow down.",
                    "limit": f"{self.rpm} requests per minute",
                    "retry_after_seconds": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        self._requests[client_ip].append(now)
        return await call_next(request)
