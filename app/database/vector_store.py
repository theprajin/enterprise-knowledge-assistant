"""
PGVector store initialization and management.

Uses langchain-postgres PGVectorStore for vector storage and retrieval
with Google Generative AI embeddings.
"""

from langchain_postgres import PGEngine, PGVectorStore
from app.config import settings
from app.embeddings.google_embeddings import get_embeddings

# Lazy connection engine initialization
_engine = None


def get_vector_engine() -> PGEngine:
    """Get or create the PGEngine singleton for vector store operations."""
    global _engine
    if _engine is None:
        _engine = PGEngine.from_connection_string(
            connection_string=settings.psycopg3_database_url
        )
    return _engine


def get_vector_store(collection_name: str | None = None) -> PGVectorStore:
    """
    Get or create a PGVectorStore instance for the given collection name.

    Args:
        collection_name: The name of the vector table/collection.
                         Defaults to settings.default_collection_name.
    """
    if collection_name is None:
        collection_name = settings.default_collection_name

    engine = get_vector_engine()
    embeddings = get_embeddings()

    # Initialize the table (handles CREATE EXTENSION vector and CREATE TABLE if not exists)
    try:
        engine.init_vectorstore_table(
            table_name=collection_name,
            vector_size=settings.embedding_dimensions,
        )
    except Exception:
        # init_vectorstore_table is not idempotent and will raise an exception
        # if the table already exists. It is safe to ignore this error.
        pass

    return PGVectorStore.create_sync(
        engine=engine,
        table_name=collection_name,
        embedding_service=embeddings,
    )
