"""
Enterprise Knowledge Assistant — FastAPI Application Entry Point.

Production-grade RAG API with:
- Versioned prompt management
- LLM output guardrails (validation, PII filtering, fallback logic)
- Multi-turn conversation context
- AI observability (latency, token usage, cost tracking)
- Rate limiting and CORS
- Structured JSON logging with request tracing
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.config import settings
from app.database.connection import get_db
from app.logging_config import setup_logging, RequestLoggingMiddleware
from app.middleware.rate_limiter import RateLimitMiddleware

# Import routers
from app.api.files import router as files_router
from app.api.vectors import router as vectors_router
from app.api.rag import router as rag_router
from app.api.prompts import router as prompts_router
from app.api.observability import router as observability_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events for startup and shutdown."""
    # ── Startup ──────────────────────────────────────────────────────
    setup_logging()
    logger.info("Enterprise Knowledge Assistant starting up...")
    logger.info(f"LLM Model: {settings.llm_model}")
    logger.info(f"Embedding Model: {settings.embedding_model}")
    logger.info(f"Rate Limit: {settings.rate_limit_rpm} RPM")
    logger.info(f"Log Level: {settings.log_level}")

    yield

    # ── Shutdown ─────────────────────────────────────────────────────
    logger.info("Enterprise Knowledge Assistant shutting down...")


# ── App Configuration ─────────────────────────────────────────────────────

app = FastAPI(
    title="Enterprise Knowledge Assistant",
    description=(
        "Production-grade RAG API for enterprise document Q&A. "
        "Features: versioned prompts, LLM guardrails, multi-turn conversations, "
        "AI observability, and multi-format document ingestion."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware (order matters: outermost first) ───────────────────────────

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate Limiting
app.add_middleware(RateLimitMiddleware)

# Request Logging
app.add_middleware(RequestLoggingMiddleware)

# ── Register Routers ─────────────────────────────────────────────────────

app.include_router(files_router, prefix="/api")
app.include_router(vectors_router, prefix="/api")
app.include_router(rag_router, prefix="/api")
app.include_router(prompts_router, prefix="/api")
app.include_router(observability_router, prefix="/api")


# ── Root & Health Endpoints ───────────────────────────────────────────────

@app.get("/")
def read_root():
    """API root — returns service info and available endpoints."""
    return {
        "service": "Enterprise Knowledge Assistant",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "files": "/api/files",
            "vectors": "/api/vectors",
            "rag": "/api/rag/query",
            "prompts": "/api/prompts",
            "observability": "/api/observability/metrics",
            "conversations": "/api/rag/sessions",
        },
    }


@app.get("/health")
def health_check():
    """Basic health check — returns service status."""
    return {
        "status": "healthy",
        "service": "enterprise-knowledge-assistant",
        "version": "2.0.0",
    }


@app.get("/db-check")
def db_check(db: Session = Depends(get_db)):
    """Verify database connectivity with a simple query."""
    try:
        result = db.execute(text("SELECT 1")).fetchone()
        if result and result[0] == 1:
            return {
                "database": "connected",
                "message": "Database connection verified successfully",
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Database query returned unexpected result",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database connection failed: {str(e)}",
        )
