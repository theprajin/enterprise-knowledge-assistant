"""
Google Generative AI embeddings configuration.

Provides a factory function for creating embedding model instances
using the centralized application settings.
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Get the Google Generative AI embeddings model instance.

    Returns:
        GoogleGenerativeAIEmbeddings configured with the model and API key
        from application settings.

    Raises:
        ValueError: If the Google API key is not configured.
    """
    if not settings.google_api_key:
        raise ValueError(
            "Google API key is not set. "
            "Please set GOOGLE_API_KEY in your .env file."
        )

    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )
