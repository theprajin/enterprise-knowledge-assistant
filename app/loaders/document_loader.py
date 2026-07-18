"""
Multi-format document loader.

Supports PDF, TXT, MD, DOCX, CSV, HTML, and JSON files using
appropriate LangChain loaders for each format.
"""

import os
import json
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
    BSHTMLLoader,
)

# Supported file formats and their descriptions
SUPPORTED_FORMATS = {
    ".pdf": "PDF documents",
    ".txt": "Plain text files",
    ".md": "Markdown files",
    ".docx": "Microsoft Word documents",
    ".csv": "Comma-separated values",
    ".html": "HTML web pages",
    ".htm": "HTML web pages",
    ".json": "JSON data files",
}


def get_supported_formats() -> dict[str, str]:
    """Get a mapping of supported file extensions to descriptions."""
    return SUPPORTED_FORMATS.copy()


def is_supported_format(filename: str) -> bool:
    """Check if a file's extension is supported."""
    _, ext = os.path.splitext(filename.lower())
    return ext in SUPPORTED_FORMATS


def _load_json(file_path: str) -> List[Document]:
    """
    Custom JSON loader that flattens JSON structures into text documents.
    
    Handles both JSON arrays (each element → one document) and
    JSON objects (entire object → one document).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    base_metadata = {"source": os.path.basename(file_path), "format": "json"}

    if isinstance(data, list):
        for i, item in enumerate(data):
            content = json.dumps(item, indent=2, ensure_ascii=False)
            metadata = {**base_metadata, "item_index": i}
            documents.append(Document(page_content=content, metadata=metadata))
    elif isinstance(data, dict):
        content = json.dumps(data, indent=2, ensure_ascii=False)
        documents.append(Document(page_content=content, metadata=base_metadata))
    else:
        content = str(data)
        documents.append(Document(page_content=content, metadata=base_metadata))

    return documents


def load_document(file_path: str) -> List[Document]:
    """
    Load a document from the local file system.
    
    Supports: PDF, TXT, MD, DOCX, CSV, HTML, JSON.
    
    Args:
        file_path: Absolute path to the file.
        
    Returns:
        List of Document objects with page_content and metadata.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
        RuntimeError: If loading fails for any other reason.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    _, ext = os.path.splitext(file_path.lower())

    if ext not in SUPPORTED_FORMATS:
        supported = ", ".join(SUPPORTED_FORMATS.keys())
        raise ValueError(
            f"Unsupported file format: '{ext}'. "
            f"Supported formats: {supported}"
        )

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext in (".txt", ".md"):
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        elif ext == ".csv":
            loader = CSVLoader(file_path, encoding="utf-8")
        elif ext in (".html", ".htm"):
            loader = BSHTMLLoader(file_path, open_encoding="utf-8")
        elif ext == ".json":
            return _load_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        documents = loader.load()
        return documents

    except (FileNotFoundError, ValueError):
        raise
    except Exception as e:
        raise RuntimeError(f"Error loading document {file_path}: {str(e)}")
