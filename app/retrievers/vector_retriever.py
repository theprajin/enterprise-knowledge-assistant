from langchain_core.vectorstores import VectorStoreRetriever
from app.database.vector_store import get_vector_store

def get_retriever(
    collection_name: str = "enterprise_knowledge",
    search_type: str = "similarity",
    k: int = 4,
    score_threshold: float = 0.5
) -> VectorStoreRetriever:
    """
    Get a LangChain retriever wrapper for the PGVector store.
    """
    vector_store = get_vector_store(collection_name)
    
    search_kwargs = {"k": k}
    if search_type == "similarity_score_threshold":
        search_kwargs["score_threshold"] = score_threshold
        
    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
