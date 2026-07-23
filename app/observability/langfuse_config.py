"""
Centralized Langfuse configuration for observability and prompt management.

Provides factory functions that gracefully handle missing configuration,
allowing the application to run without Langfuse when credentials are not set.
"""

import os
import logging
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Check if Langfuse is configured
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip("\"'")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "").strip("\"'")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "http://langfuse-web:3000").strip(
    "\"'"
)

_langfuse_available = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)

if not _langfuse_available:
    logger.warning(
        "Langfuse credentials not found (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY). "
        "Observability and prompt management are disabled. "
        "Set these in your .env file after creating a project in the Langfuse UI."
    )


def get_langfuse_handler():
    """
    Return a LangChain CallbackHandler for automatic tracing.

    Returns None if Langfuse is not configured, allowing the app
    to function without observability.
    """
    if not _langfuse_available:
        return None

    try:
        from langfuse.langchain import CallbackHandler

        # Credentials are read from LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL env vars
        handler = CallbackHandler()
        return handler
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse CallbackHandler: {e}")
        return None


def get_langfuse_client():
    """
    Return a Langfuse client instance for prompt management and other SDK features.

    Returns None if Langfuse is not configured.
    """
    if not _langfuse_available:
        return None

    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_BASE_URL,
        )
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse client: {e}")
        return None


def is_langfuse_configured() -> bool:
    """Check if Langfuse credentials are present in the environment."""
    return _langfuse_available


def check_langfuse_health() -> dict:
    """
    Check if the Langfuse server is reachable and the credentials are valid.

    Returns a dict with status information.
    """
    if not _langfuse_available:
        return {
            "status": "not_configured",
            "message": "Langfuse credentials not set in environment",
        }

    try:
        client = get_langfuse_client()
        if client is None:
            return {
                "status": "error",
                "message": "Failed to initialize Langfuse client",
            }

        # Attempt an auth check by fetching prompts (lightweight call)
        client.auth_check()
        return {
            "status": "connected",
            "host": LANGFUSE_BASE_URL,
            "message": "Langfuse is reachable and credentials are valid",
        }
    except Exception as e:
        return {
            "status": "error",
            "host": LANGFUSE_BASE_URL,
            "message": f"Langfuse health check failed: {str(e)}",
        }
