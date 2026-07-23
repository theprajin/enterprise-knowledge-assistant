import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.retrievers.vector_retriever import get_retriever
from app.observability.langfuse_config import get_langfuse_handler, get_langfuse_client
from langchain_groq import ChatGroq

load_dotenv()

logger = logging.getLogger(__name__)

# Resolve potential case sensitivity issues in environment variables
google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("Google_API_KEY")
if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key

# Default system prompt for RAG (used as fallback when Langfuse prompt is unavailable)
RAG_SYSTEM_PROMPT = """You are an Enterprise Knowledge Assistant. Answer the user's question using only the provided context. 
If you do not know the answer or if the context does not contain enough information, state clearly that you cannot answer based on the provided documents. Do not try to make up an answer.

Context:
{context}
"""


def _get_prompt_from_langfuse():
    """
    Attempt to fetch the RAG system prompt from Langfuse for versioned prompt management.

    Returns a tuple of (langchain_prompt, langfuse_prompt_object) if found,
    or (None, None) if Langfuse is not configured or the prompt doesn't exist.
    """
    try:
        client = get_langfuse_client()
        if client is None:
            return None, None

        langfuse_prompt = client.get_prompt("rag-system-prompt", type="chat")
        langchain_prompt = ChatPromptTemplate.from_messages(
            langfuse_prompt.get_langchain_prompt()
        )
        logger.info(
            f"Using Langfuse-managed prompt 'rag-system-prompt' "
            f"(version: {langfuse_prompt.version})"
        )
        return langchain_prompt, langfuse_prompt
    except Exception as e:
        logger.debug(
            f"Langfuse prompt 'rag-system-prompt' not found or unavailable: {e}. "
            f"Falling back to hardcoded prompt."
        )
        return None, None


def query_rag(
    query: str, collection_name: str = "enterprise_knowledge", k: int = 4
) -> Dict[str, Any]:
    """
    Execute a RAG query: retrieve context, prompt Gemini, and return the answer along with sources.

    Automatically traces the execution in Langfuse when configured. Supports
    versioned prompts from Langfuse with fallback to the hardcoded default.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError(
            "Google API key is not set. Please set GOOGLE_API_KEY or Google_API_KEY in your environment."
        )

    # 1. Retrieve relevant documents
    retriever = get_retriever(collection_name=collection_name, k=k)
    docs = retriever.invoke(query)

    # 2. Format context from documents
    context_chunks = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", i + 1)
        context_chunks.append(f"Source: {source} (Page {page}):\n{doc.page_content}")

    context = "\n\n".join(context_chunks)

    # 3. Try to fetch versioned prompt from Langfuse, fall back to hardcoded default
    langfuse_prompt_obj = None
    langchain_prompt, langfuse_prompt_obj = _get_prompt_from_langfuse()

    if langchain_prompt is None:
        # Use the hardcoded default prompt
        langchain_prompt = ChatPromptTemplate.from_messages(
            [("system", RAG_SYSTEM_PROMPT), ("human", "{question}")]
        )

    # 4. Create LLM
    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-1.5-flash",
    #     google_api_key=os.getenv("GOOGLE_API_KEY"),
    #     temperature=0.0
    # )

    llm = ChatGroq(model="llama-3.1-8b-instant")

    # 5. Execute Chain with optional Langfuse tracing
    chain = langchain_prompt | llm | StrOutputParser()

    # Build invocation config with Langfuse callback if available
    invoke_config = {}
    langfuse_handler = get_langfuse_handler()
    if langfuse_handler is not None:
        invoke_config["callbacks"] = [langfuse_handler]

        # Link the Langfuse prompt version to the trace for tracking
        if langfuse_prompt_obj is not None:
            invoke_config["metadata"] = {"langfusePrompt": langfuse_prompt_obj}

    answer = chain.invoke(
        {"context": context, "question": query},
        config=invoke_config if invoke_config else None,
    )

    # 6. Extract sources details
    sources = []
    seen_sources = set()
    for doc in docs:
        source_name = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 1)
        source_key = f"{source_name}_page_{page}"

        # Avoid duplicate pages in the source list summary
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append(
                {
                    "source": source_name,
                    "page": page,
                    "content_preview": (
                        doc.page_content[:150] + "..."
                        if len(doc.page_content) > 150
                        else doc.page_content
                    ),
                }
            )

    return {"query": query, "answer": answer, "sources": sources}
