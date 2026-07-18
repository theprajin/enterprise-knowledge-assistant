"""
Centralized application configuration using Pydantic BaseSettings.

All environment variables are loaded and validated here. Other modules should
import `settings` from this module instead of calling os.getenv() directly.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    # ── Database ──────────────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql://postgres:postgres@db:5432/knowledge_assistant",
        description="PostgreSQL connection URL"
    )
    postgres_user: str = Field(default="postgres")
    postgres_password: str = Field(default="postgres")
    postgres_db: str = Field(default="knowledge_assistant")

    # ── Google AI ─────────────────────────────────────────────────────────
    google_api_key: str = Field(
        default="",
        description="Google API key for Gemini and embedding models"
    )

    # ── LLM Configuration ─────────────────────────────────────────────────
    llm_model: str = Field(
        default="gemini-2.0-flash",
        description="Google Generative AI model name"
    )
    llm_temperature: float = Field(
        default=0.0, ge=0.0, le=2.0,
        description="LLM temperature for response generation"
    )
    embedding_model: str = Field(
        default="models/text-embedding-004",
        description="Embedding model name"
    )
    embedding_dimensions: int = Field(
        default=768,
        description="Embedding vector dimensions"
    )

    # ── RAG Defaults ──────────────────────────────────────────────────────
    default_chunk_size: int = Field(
        default=1000, ge=100, le=10000,
        description="Default chunk size for document splitting"
    )
    default_chunk_overlap: int = Field(
        default=200, ge=0, le=2000,
        description="Default chunk overlap for document splitting"
    )
    default_collection_name: str = Field(
        default="enterprise_knowledge",
        description="Default pgvector collection/table name"
    )
    default_retrieval_k: int = Field(
        default=4, ge=1, le=20,
        description="Default number of documents to retrieve"
    )

    # ── File Upload ───────────────────────────────────────────────────────
    upload_dir: str = Field(
        default="uploads",
        description="Directory for uploaded files (relative to project root)"
    )
    max_upload_size_mb: int = Field(
        default=50, ge=1, le=500,
        description="Maximum upload file size in MB"
    )

    # ── Rate Limiting ─────────────────────────────────────────────────────
    rate_limit_rpm: int = Field(
        default=60, ge=1,
        description="Rate limit: maximum requests per minute per IP"
    )

    # ── Logging ───────────────────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )

    # ── CORS ──────────────────────────────────────────────────────────────
    cors_origins: str = Field(
        default="*",
        description="Comma-separated list of allowed CORS origins"
    )

    @field_validator("google_api_key", mode="before")
    @classmethod
    def resolve_google_api_key(cls, v, info):
        """Accept both GOOGLE_API_KEY and Google_API_KEY for compatibility."""
        import os
        if not v:
            v = os.getenv("Google_API_KEY", "")
        return v

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper()
        if v not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def max_upload_size_bytes(self) -> int:
        """Get max upload size in bytes."""
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def psycopg3_database_url(self) -> str:
        """Get database URL with psycopg3 driver prefix for langchain-postgres."""
        url = self.database_url
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+psycopg://", 1)
        return url

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings singleton."""
    return Settings()


# Module-level convenience instance
settings = get_settings()
