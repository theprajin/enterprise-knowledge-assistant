"""
Structured logging configuration for production observability.

Provides JSON-formatted logs with request tracing (request_id) and
a FastAPI middleware for automatic request/response logging.
"""

import logging
import json
import time
import uuid
from contextvars import ContextVar
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from app.config import settings

# Context variable for request-scoped tracing
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "message": record.getMessage(),
            "request_id": request_id_ctx.get("-"),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        # Include any extra fields attached to the log record
        for key in ("latency_ms", "status_code", "method", "path", "client_ip"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        return json.dumps(log_entry)


def setup_logging() -> None:
    """Configure structured JSON logging for the application."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level, logging.INFO))

    # Clear existing handlers to avoid duplicate logs
    root_logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    root_logger.addHandler(handler)

    # Suppress noisy third-party loggers
    for noisy_logger in ("uvicorn.access", "httpcore", "httpx"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware that:
    - Generates a unique request_id per request
    - Logs request start and completion with latency
    - Sets request_id in context for downstream log correlation
    """

    async def dispatch(self, request: Request, call_next):
        req_id = str(uuid.uuid4())[:8]
        request_id_ctx.set(req_id)

        # Attach request_id to response headers for client-side tracing
        request.state.request_id = req_id

        logger = logging.getLogger("app.request")
        logger.info(
            f"→ {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": str(request.url.path),
                "client_ip": request.client.host if request.client else "unknown",
            },
        )

        start_time = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled exception during request processing")
            raise

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        logger.info(
            f"← {request.method} {request.url.path} → {response.status_code} ({latency_ms}ms)",
            extra={
                "method": request.method,
                "path": str(request.url.path),
                "status_code": response.status_code,
                "latency_ms": latency_ms,
            },
        )

        response.headers["X-Request-ID"] = req_id
        return response
