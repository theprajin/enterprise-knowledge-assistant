"""
File management API endpoints.

Handles file upload with validation, listing, loading/parsing,
and deletion of uploaded documents.
"""

import os
import shutil
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.config import settings
from app.loaders import load_document, get_supported_formats, is_supported_format

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/files",
    tags=["files"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIR = os.path.join(BASE_DIR, settings.upload_dir)
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.get("/")
async def list_files():
    """
    List all uploaded files with metadata.
    
    Returns filename, size, content type, and modification time for each file.
    """
    files = []
    if os.path.exists(UPLOAD_DIR):
        for filename in sorted(os.listdir(UPLOAD_DIR)):
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                stat = os.stat(file_path)
                _, ext = os.path.splitext(filename.lower())
                files.append({
                    "filename": filename,
                    "size_bytes": stat.st_size,
                    "size_human": _format_size(stat.st_size),
                    "format": ext,
                    "supported": is_supported_format(filename),
                    "modified_at": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                })

    return {
        "files": files,
        "total": len(files),
        "upload_dir": UPLOAD_DIR,
    }


@router.get("/formats")
async def list_supported_formats():
    """List all supported document formats for upload and ingestion."""
    return {
        "formats": get_supported_formats(),
        "total": len(get_supported_formats()),
    }


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file to the server.
    
    Validates file format and size before saving.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file was selected")

    # Validate file format
    if not is_supported_format(file.filename):
        supported = ", ".join(get_supported_formats().keys())
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported: {supported}",
        )

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        # Read file content and check size
        content = await file.read()
        if len(content) > settings.max_upload_size_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.max_upload_size_mb}MB",
            )

        with open(file_path, "wb") as buffer:
            buffer.write(content)

        logger.info(f"File uploaded: {file.filename} ({len(content)} bytes)")

        return JSONResponse(
            status_code=201,
            content={
                "message": "File uploaded successfully",
                "filename": file.filename,
                "size_bytes": len(content),
                "size_human": _format_size(len(content)),
                "content_type": file.content_type,
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.exception(f"Failed to upload file: {file.filename}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {str(e)}",
        )


@router.get("/load/{filename}")
async def load_uploaded_file(filename: str):
    """
    Load and parse an uploaded file using the document loader.
    
    Returns a summary of each page/chunk including metadata and content preview.
    """
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"File '{filename}' not found in uploads directory",
        )

    try:
        documents = load_document(file_path)

        pages_summary = []
        for i, doc in enumerate(documents):
            pages_summary.append({
                "page": i + 1,
                "metadata": doc.metadata,
                "content_preview": doc.page_content[:200],
                "content_length": len(doc.page_content),
            })

        return {
            "filename": filename,
            "total_pages_or_chunks": len(documents),
            "pages": pages_summary,
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception(f"Failed to load file: {filename}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load and parse file: {str(e)}",
        )


@router.delete("/{filename}")
async def delete_file(filename: str):
    """Delete an uploaded file from the server."""
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"File '{filename}' not found",
        )

    try:
        os.remove(file_path)
        logger.info(f"File deleted: {filename}")
        return {"message": f"File '{filename}' deleted successfully"}
    except Exception as e:
        logger.exception(f"Failed to delete file: {filename}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete file: {str(e)}",
        )


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
