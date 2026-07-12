import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def load_document(file_path: str) -> List[Document]:
    """
    Load a document from the local file system.
    Supports PDF, TXT, MD files.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    _, ext = os.path.splitext(file_path.lower())

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".txt", ".md"]:
        # TextLoader handles txt, md, etc.
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported formats are PDF, TXT, MD.")

    try:
        documents = loader.load()
        return documents
    except Exception as e:
        raise RuntimeError(f"Error loading document {file_path}: {str(e)}")
