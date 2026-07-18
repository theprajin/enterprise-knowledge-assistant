"""
RAG (Retrieval-Augmented Generation) service.

Orchestrates the full RAG pipeline:
1. Question reformulation (for multi-turn conversations)
2. Document retrieval from pgvector
3. Context formatting
4. Prompt selection (versioned)
5. LLM invocation with token tracking
6. Guardrail validation
7. Metrics recording

Uses the LangChain ecosystem throughout — no LangGraph.
"""

import time
import uuid
import logging
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import settings
from app.retrievers.vector_retriever import get_retriever
from app.prompts.registry import get_prompt_registry
from app.guardrails.fallback import apply_guardrails
from app.services.conversation import get_conversation_manager
from app.observability.tracker import (
    QueryMetrics,
    get_metrics_tracker,
    estimate_cost,
)
from app.observability.callbacks import TokenUsageCallbackHandler

logger = logging.getLogger(__name__)


def _get_llm(callbacks: list | None = None) -> ChatGoogleGenerativeAI:
    """Create a configured LLM instance."""
    if not settings.google_api_key:
        raise ValueError(
            "Google API key is not set. "
            "Please set GOOGLE_API_KEY in your .env file."
        )

    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.google_api_key,
        temperature=settings.llm_temperature,
        callbacks=callbacks,
    )


def _reformulate_question(
    question: str, chat_history: str, callbacks: list | None = None
) -> str:
    """
    Reformulate a follow-up question into a standalone question
    using conversation history context.
    """
    registry = get_prompt_registry()
    reformulation_prompt = registry.get("question_reformulation")

    prompt = ChatPromptTemplate.from_messages([
        ("system", reformulation_prompt.system_template),
        ("human", reformulation_prompt.human_template),
    ])

    llm = _get_llm(callbacks=callbacks)
    chain = prompt | llm | StrOutputParser()

    reformulated = chain.invoke({
        "chat_history": chat_history,
        "question": question,
    })

    logger.info(f"Question reformulated: '{question}' → '{reformulated.strip()}'")
    return reformulated.strip()


def query_rag(
    query: str,
    collection_name: str | None = None,
    k: int | None = None,
    prompt_name: str = "rag_default",
    prompt_version: int | None = None,
    conversation_id: str | None = None,
) -> dict:
    """
    Execute a full RAG query with all production features.
    
    Args:
        query: The user's question.
        collection_name: pgvector collection to search.
        k: Number of documents to retrieve.
        prompt_name: Name of the prompt template to use.
        prompt_version: Specific prompt version (None = latest).
        conversation_id: Optional conversation session ID for multi-turn.
        
    Returns:
        Dict with answer, sources, metrics, and guardrail status.
    """
    query_id = str(uuid.uuid4())[:12]
    start_time = time.perf_counter()

    if collection_name is None:
        collection_name = settings.default_collection_name
    if k is None:
        k = settings.default_retrieval_k

    tracker = get_metrics_tracker()
    token_callback = TokenUsageCallbackHandler()

    try:
        # ── Step 1: Handle conversation context ──────────────────────
        effective_query = query
        conversation_mgr = get_conversation_manager()
        session = None

        if conversation_id:
            session = conversation_mgr.get_session(conversation_id)
            if session and session.messages:
                # Use conversational prompt variant
                if prompt_name == "rag_default":
                    prompt_name = "rag_conversational"

                # Reformulate the question for better retrieval
                chat_history = session.get_history_text(max_turns=5)
                effective_query = _reformulate_question(
                    query, chat_history, callbacks=[token_callback]
                )

        # ── Step 2: Retrieve relevant documents ──────────────────────
        retrieval_start = time.perf_counter()

        retriever = get_retriever(collection_name=collection_name, k=k)
        docs = retriever.invoke(effective_query)

        retrieval_latency = round(
            (time.perf_counter() - retrieval_start) * 1000, 2
        )

        # ── Step 3: Format context from documents ────────────────────
        context_chunks = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", i + 1)
            context_chunks.append(
                f"Source: {source} (Page {page}):\n{doc.page_content}"
            )

        context = "\n\n".join(context_chunks)
        context_provided = len(docs) > 0

        # ── Step 4: Select prompt template ───────────────────────────
        registry = get_prompt_registry()
        selected_prompt = registry.get(prompt_name, prompt_version)

        prompt = ChatPromptTemplate.from_messages([
            ("system", selected_prompt.system_template),
            ("human", selected_prompt.human_template),
        ])

        # ── Step 5: Invoke LLM with token tracking ──────────────────
        llm_start = time.perf_counter()

        llm = _get_llm(callbacks=[token_callback])
        chain = prompt | llm | StrOutputParser()
        raw_answer = chain.invoke({"context": context, "question": query})

        llm_latency = round((time.perf_counter() - llm_start) * 1000, 2)

        # ── Step 6: Apply guardrails ─────────────────────────────────
        guardrail_result = apply_guardrails(
            raw_answer, context_provided=context_provided
        )
        answer = guardrail_result.filtered_response

        # ── Step 7: Update conversation history ──────────────────────
        if conversation_id and session:
            conversation_mgr.add_exchange(conversation_id, query, answer)

        # ── Step 8: Extract sources ──────────────────────────────────
        sources = []
        seen_sources = set()
        for doc in docs:
            source_name = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", 1)
            source_key = f"{source_name}_page_{page}"

            if source_key not in seen_sources:
                seen_sources.add(source_key)
                sources.append({
                    "source": source_name,
                    "page": page,
                    "content_preview": (
                        doc.page_content[:150] + "..."
                        if len(doc.page_content) > 150
                        else doc.page_content
                    ),
                })

        # ── Step 9: Record metrics ───────────────────────────────────
        total_latency = round((time.perf_counter() - start_time) * 1000, 2)
        usage = token_callback.get_usage()

        metrics = QueryMetrics(
            query_id=query_id,
            model=settings.llm_model,
            prompt_name=prompt_name,
            prompt_version=selected_prompt.version,
            latency_ms=total_latency,
            retrieval_latency_ms=retrieval_latency,
            llm_latency_ms=llm_latency,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            total_tokens=usage["total_tokens"],
            estimated_cost_usd=estimate_cost(
                settings.llm_model,
                usage["input_tokens"],
                usage["output_tokens"],
            ),
            retrieval_count=len(docs),
            conversation_id=conversation_id,
            guardrails_passed=guardrail_result.passed,
            guardrail_violations=len(guardrail_result.violations),
        )
        tracker.record(metrics)

        # ── Step 10: Build response ──────────────────────────────────
        return {
            "query_id": query_id,
            "query": query,
            "answer": answer,
            "sources": sources,
            "prompt": {
                "name": prompt_name,
                "version": selected_prompt.version,
            },
            "guardrails": guardrail_result.to_dict(),
            "metrics": {
                "latency_ms": total_latency,
                "retrieval_latency_ms": retrieval_latency,
                "llm_latency_ms": llm_latency,
                "tokens_used": usage["total_tokens"],
                "estimated_cost_usd": metrics.estimated_cost_usd,
            },
            "conversation_id": conversation_id,
        }

    except Exception as e:
        tracker.record_error()
        logger.exception(f"RAG query failed: {e}")
        raise
