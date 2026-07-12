import os
from fastapi import APIRouter, HTTPException, Query
from app.loaders import load_document
from app.splitters.text_splitter import split_documents
from app.database.vector_store import get_vector_store

router = APIRouter(
    prefix="/vectors",
    tags=["vectors"]
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

@router.post("/ingest/{filename}")
async def ingest_document(filename: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Load an uploaded document, split it into chunks, and ingest it into the pgvector store.
    """
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404, 
            detail=f"File '{filename}' not found in uploads directory. Please upload it first."
        )
        
    try:
        # Load the document
        raw_docs = load_document(file_path)
        
        # Split the document into chunks
        chunked_docs = split_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Add metadata source
        for doc in chunked_docs:
            doc.metadata["source"] = filename
        
        # Ingest into vector store
        vector_store = get_vector_store()
        vector_store.add_documents(chunked_docs)
        
        return {
            "message": "Document ingested successfully",
            "filename": filename,
            "chunks_created": len(chunked_docs)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )

@router.get("/search")
async def search_vectors(
    query: str = Query(..., description="The query to search for"), 
    k: int = Query(4, description="Number of results to return")
):
    """
    Search the vector store for documents similar to the query.
    """
    try:
        vector_store = get_vector_store()
        results = vector_store.similarity_search(query, k=k)
        
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
            
        return {
            "query": query,
            "results_count": len(formatted_results),
            "results": formatted_results
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )
