import os
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.retrievers.vector_retriever import get_retriever

load_dotenv()

# Resolve potential case sensitivity issues in environment variables
google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("Google_API_KEY")
if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key

# System prompt for RAG
RAG_SYSTEM_PROMPT = """You are an Enterprise Knowledge Assistant. Answer the user's question using only the provided context. 
If you do not know the answer or if the context does not contain enough information, state clearly that you cannot answer based on the provided documents. Do not try to make up an answer.

Context:
{context}
"""

def query_rag(query: str, collection_name: str = "enterprise_knowledge", k: int = 4) -> Dict[str, Any]:
    """
    Execute a RAG query: retrieve context, prompt Gemini, and return the answer along with sources.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("Google API key is not set. Please set GOOGLE_API_KEY or Google_API_KEY in your environment.")

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
    
    # 3. Create prompt and LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "{question}")
    ])
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.0
    )
    
    # 4. Execute Chain
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})
    
    # 5. Extract sources details
    sources = []
    seen_sources = set()
    for doc in docs:
        source_name = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 1)
        source_key = f"{source_name}_page_{page}"
        
        # Avoid duplicate pages in the source list summary
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append({
                "source": source_name,
                "page": page,
                "content_preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
            })
        
    return {
        "query": query,
        "answer": answer,
        "sources": sources
    }
