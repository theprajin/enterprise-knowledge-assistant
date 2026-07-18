"""
RAG API endpoints.

Provides query execution with prompt selection, conversation management,
and full observability metadata in responses.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.rag.rag_service import query_rag
from app.services.conversation import get_conversation_manager

router = APIRouter(
    prefix="/rag",
    tags=["rag"],
)


# ── Request / Response Models ─────────────────────────────────────────────

class RAGQueryRequest(BaseModel):
    """Request body for RAG query execution."""

    query: str = Field(
        ..., description="The question to answer based on ingested documents"
    )
    k: int = Field(
        4, ge=1, le=20,
        description="Number of source chunks to retrieve"
    )
    prompt_name: str = Field(
        "rag_default",
        description="Prompt template name (rag_default, rag_concise, rag_detailed, rag_cot)"
    )
    prompt_version: Optional[int] = Field(
        None,
        description="Specific prompt version (None = latest)"
    )
    conversation_id: Optional[str] = Field(
        None,
        description="Conversation session ID for multi-turn context"
    )


# ── RAG Query Endpoint ───────────────────────────────────────────────────

@router.post("/query")
async def execute_rag_query(request: RAGQueryRequest):
    """
    Execute a RAG query with full production features.
    
    Retrieves context from pgvector, selects a versioned prompt template,
    invokes the LLM, applies guardrails, and returns the answer with
    source citations, metrics, and guardrail status.
    """
    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query string cannot be empty"
        )

    try:
        result = query_rag(
            query=request.query,
            k=request.k,
            prompt_name=request.prompt_name,
            prompt_version=request.prompt_version,
            conversation_id=request.conversation_id,
        )
        return result
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG query execution failed: {str(e)}"
        )


# ── Conversation Session Endpoints ───────────────────────────────────────

@router.post("/sessions")
async def create_conversation_session():
    """Create a new conversation session for multi-turn RAG queries."""
    manager = get_conversation_manager()
    session = manager.create_session()
    return {
        "session_id": session.session_id,
        "created_at": session.created_at,
        "message": "Conversation session created. Pass this session_id in your RAG queries for multi-turn context.",
    }


@router.get("/sessions")
async def list_conversation_sessions():
    """List all active conversation sessions."""
    manager = get_conversation_manager()
    sessions = manager.list_sessions()
    return {
        "sessions": sessions,
        "total": len(sessions),
    }


@router.get("/sessions/{session_id}")
async def get_conversation_history(session_id: str):
    """Get the full message history for a conversation session."""
    manager = get_conversation_manager()
    session = manager.get_session(session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found"
        )

    return session.to_dict()


@router.delete("/sessions/{session_id}")
async def delete_conversation_session(session_id: str):
    """Delete a conversation session and its history."""
    manager = get_conversation_manager()
    deleted = manager.clear_session(session_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found"
        )

    return {"message": f"Session '{session_id}' deleted successfully"}
