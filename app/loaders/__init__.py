# Loaders package initialization
from app.loaders.document_loader import load_document, get_supported_formats, is_supported_format

__all__ = ["load_document", "get_supported_formats", "is_supported_format"]
