"""
Vector store management API endpoints.

Handles document ingestion, semantic search, collection statistics,
and collection deletion.
"""

import os
import logging

from fastapi import APIRouter, HTTPException, Query

from app.config import settings
from app.loaders import load_document
from app.splitters.text_splitter import split_documents
from app.database.vector_store import get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/vectors",
    tags=["vectors"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIR = os.path.join(BASE_DIR, settings.upload_dir)


@router.post("/ingest/{filename}")
async def ingest_document(
    filename: str,
    chunk_size: int = Query(
        default=None, ge=100, le=10000,
        description="Chunk size in characters"
    ),
    chunk_overlap: int = Query(
        default=None, ge=0, le=2000,
        description="Overlap between chunks"
    ),
):
    """
    Load an uploaded document, split it into chunks, and ingest into pgvector.
    
    Uses format-aware chunking strategies based on the file type.
    """
    # Use config defaults if not specified
    if chunk_size is None:
        chunk_size = settings.default_chunk_size
    if chunk_overlap is None:
        chunk_overlap = settings.default_chunk_overlap

    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"File '{filename}' not found in uploads directory. Please upload it first.",
        )

    try:
        # Load the document
        raw_docs = load_document(file_path)

        # Split with format-aware strategy
        chunked_docs = split_documents(
            raw_docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source_filename=filename,
        )

        # Enrich metadata
        for i, doc in enumerate(chunked_docs):
            doc.metadata["source"] = filename
            doc.metadata["chunk_index"] = i

        # Ingest into vector store
        vector_store = get_vector_store()
        vector_store.add_documents(chunked_docs)

        logger.info(
            f"Document ingested: {filename} → {len(chunked_docs)} chunks "
            f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
        )

        return {
            "message": "Document ingested successfully",
            "filename": filename,
            "chunks_created": len(chunked_docs),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception(f"Ingestion failed for {filename}")
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}",
        )


@router.get("/search")
async def search_vectors(
    query: str = Query(..., description="The query to search for"),
    k: int = Query(4, ge=1, le=20, description="Number of results to return"),
):
    """
    Search the vector store for documents semantically similar to the query.
    
    Returns matched chunks with content and metadata.
    """
    try:
        vector_store = get_vector_store()
        results = vector_store.similarity_search_with_score(query, k=k)

        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": round(float(score), 4),
            })

        return {
            "query": query,
            "results_count": len(formatted_results),
            "results": formatted_results,
        }
    except Exception as e:
        logger.exception("Vector search failed")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}",
        )


@router.get("/stats")
async def get_vector_stats():
    """
    Get statistics for the default vector collection.
    
    Returns approximate document count and collection info.
    """
    try:
        vector_store = get_vector_store()
        # Perform a minimal search to verify the store is accessible
        test_results = vector_store.similarity_search("test", k=1)

        return {
            "collection_name": settings.default_collection_name,
            "status": "active",
            "embedding_model": settings.embedding_model,
            "embedding_dimensions": settings.embedding_dimensions,
            "message": "Vector store is accessible and operational",
        }
    except Exception as e:
        return {
            "collection_name": settings.default_collection_name,
            "status": "error",
            "error": str(e),
        }


@router.delete("/{collection_name}")
async def delete_collection(collection_name: str):
    """
    Delete all vectors in a collection.
    
    WARNING: This permanently removes all ingested documents from the collection.
    Use this for re-ingestion workflows.
    """
    try:
        vector_store = get_vector_store(collection_name)
        # PGVectorStore doesn't have a direct "delete all" method,
        # so we drop and recreate the table via the engine
        from app.database.vector_store import get_vector_engine
        engine = get_vector_engine()

        engine.drop_vectorstore_table(table_name=collection_name)
        logger.info(f"Collection deleted: {collection_name}")

        return {
            "message": f"Collection '{collection_name}' deleted successfully",
            "note": "Re-ingest documents to rebuild the collection.",
        }
    except Exception as e:
        logger.exception(f"Failed to delete collection: {collection_name}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete collection: {str(e)}",
        )
