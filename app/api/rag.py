from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.rag.rag_service import query_rag

router = APIRouter(
    prefix="/rag",
    tags=["rag"]
)

class RAGQueryRequest(BaseModel):
    query: str = Field(..., description="The query/question to answer based on documents")
    k: int = Field(4, ge=1, le=10, description="Number of source chunks to retrieve")

@router.post("/query")
async def execute_rag_query(request: RAGQueryRequest):
    """
    Perform a RAG query: Retrieves context from pgvector and runs the Gemini LLM to answer the question.
    """
    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query string cannot be empty"
        )
        
    try:
        result = query_rag(query=request.query, k=request.k)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG query execution failed: {str(e)}"
        )
