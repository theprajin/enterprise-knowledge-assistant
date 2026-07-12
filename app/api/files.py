import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.loaders import load_document

router = APIRouter(
    prefix="/files",
    tags=["files"]
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file to the server.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file was selected")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return JSONResponse(
            status_code=201,
            content={
                "message": "File uploaded successfully",
                "filename": file.filename,
                "saved_path": file_path,
                "content_type": file.content_type
            }
        )
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {str(e)}"
        )

@router.get("/load/{filename}")
async def load_uploaded_file(filename: str):
    """
    Load and parse an uploaded file using the document loader.
    """
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404, 
            detail=f"File '{filename}' not found in uploads directory"
        )
    
    try:
        documents = load_document(file_path)
        
        # Prepare a summarized representation of the loaded pages/documents
        pages_summary = []
        for i, doc in enumerate(documents):
            pages_summary.append({
                "page": i + 1,
                "metadata": doc.metadata,
                "content_preview": doc.page_content[:200]  # Show the first 200 characters
            })
            
        return {
            "filename": filename,
            "total_pages_or_chunks": len(documents),
            "pages": pages_summary
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load and parse file: {str(e)}"
        )

