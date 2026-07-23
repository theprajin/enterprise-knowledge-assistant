import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.database.connection import get_db
from app.api.files import router as files_router
from app.api.vectors import router as vectors_router
from app.api.rag import router as rag_router
from app.observability.langfuse_config import (
    is_langfuse_configured,
    check_langfuse_health,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    # Startup
    if is_langfuse_configured():
        logger.info("Langfuse is configured — observability and prompt management are enabled.")
        health = check_langfuse_health()
        if health["status"] == "connected":
            logger.info(f"Langfuse connection verified: {health['host']}")
        else:
            logger.warning(f"Langfuse is configured but not reachable: {health['message']}")
    else:
        logger.info(
            "Langfuse is not configured — running without observability. "
            "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env to enable."
        )

    yield  # Application is running

    # Shutdown (nothing to clean up currently)


app = FastAPI(
    title="Enterprise Knowledge Assistant",
    description="Backend API for Enterprise Knowledge Assistant",
    version="1.0.0",
    lifespan=lifespan,
)

# Register routers
app.include_router(files_router, prefix="/api")
app.include_router(vectors_router, prefix="/api")
app.include_router(rag_router, prefix="/api")


@app.get("/")
def read_root():
    return {
        "message": "Welcome to Enterprise Knowledge Assistant API",
        "status": "running"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/langfuse-health")
def langfuse_health():
    """
    Check the Langfuse connection status.
    Returns configuration and connectivity information.
    """
    return check_langfuse_health()

@app.get("/db-check")
def db_check(db: Session = Depends(get_db)):
    try:
        # Run a simple query to verify database connection
        result = db.execute(text("SELECT 1")).fetchone()
        if result and result[0] == 1:
            return {
                "database": "connected",
                "message": "Database connection verified successfully"
            }
        else:
            raise HTTPException(
                status_code=500, 
                detail="Database query returned unexpected result"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database connection failed: {str(e)}"
        )
