"""
Format-aware text splitting strategies.

Selects the appropriate text splitter based on file format to produce
higher-quality chunks that respect document structure.
"""

import os
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    HTMLHeaderTextSplitter,
)


def get_splitter(
    file_type: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> RecursiveCharacterTextSplitter:
    """
    Factory function that returns the appropriate splitter for a file type.
    
    Args:
        file_type: File extension (e.g., ".md", ".html", ".pdf").
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.
        
    Returns:
        A configured text splitter instance.
    """
    file_type = file_type.lower()

    if file_type == ".md":
        return MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        # Default: RecursiveCharacterTextSplitter works well for most formats
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    source_filename: str | None = None,
) -> List[Document]:
    """
    Split a list of documents into smaller chunks using format-aware splitting.
    
    Args:
        documents: List of Document objects to split.
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.
        source_filename: Optional filename to determine the splitter type.
        
    Returns:
        List of chunked Document objects.
    """
    # Determine file type from source metadata or provided filename
    file_type = ".txt"  # default
    if source_filename:
        _, file_type = os.path.splitext(source_filename.lower())
    elif documents and documents[0].metadata.get("source"):
        source = documents[0].metadata["source"]
        _, file_type = os.path.splitext(source.lower())

    splitter = get_splitter(file_type, chunk_size, chunk_overlap)
    return splitter.split_documents(documents)


def split_html_by_headers(
    html_content: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Split HTML content by header tags for structure-aware chunking.
    
    Useful for HTML documents where sections are delineated by h1-h3 tags.
    
    Args:
        html_content: Raw HTML string.
        chunk_size: Maximum chunk size for secondary splitting.
        chunk_overlap: Overlap for secondary splitting.
        
    Returns:
        List of Document objects split by HTML headers.
    """
    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
    ]

    html_splitter = HTMLHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    header_splits = html_splitter.split_text(html_content)

    # Secondary split for chunks that are still too large
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return text_splitter.split_documents(header_splits)
