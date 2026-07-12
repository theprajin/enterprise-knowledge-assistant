import os
from dotenv import load_dotenv
from langchain_postgres import PGEngine, PGVectorStore
from app.embeddings.google_embeddings import get_embeddings

load_dotenv()

# Get database connection URL and ensure it uses psycopg3 driver
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:postgres@db:5432/knowledge_assistant"
)

# langchain-postgres requires postgresql+psycopg:// (psycopg3)
if DATABASE_URL.startswith("postgresql://"):
    CONNECTION_URI = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
else:
    CONNECTION_URI = DATABASE_URL

# Lazy connection engine initialization
_engine = None

def get_vector_engine() -> PGEngine:
    global _engine
    if _engine is None:
        _engine = PGEngine.from_connection_string(connection_string=CONNECTION_URI)
    return _engine

def get_vector_store(collection_name: str = "enterprise_knowledge") -> PGVectorStore:
    """
    Get or create a PGVectorStore instance for the given collection name.
    """
    engine = get_vector_engine()
    embeddings = get_embeddings()
    
    # Initialize the table (handles CREATE EXTENSION vector and CREATE TABLE if not exists)
    # text-embedding-004 produces embeddings of size 768.
    try:
        engine.init_vectorstore_table(
            table_name=collection_name,
            vector_size=768
        )
    except Exception:
        # init_vectorstore_table is not idempotent and will raise an exception if the table already exists.
        # It is safe to ignore this error as the table is already successfully created.
        pass
        
    return PGVectorStore.create_sync(
        engine=engine,
        table_name=collection_name,
        embedding_service=embeddings
    )
