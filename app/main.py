from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.database import get_db
from app.api.files import router as files_router

app = FastAPI(
    title="Enterprise Knowledge Assistant",
    description="Backend API for Enterprise Knowledge Assistant",
    version="1.0.0"
)

# Register routers
app.include_router(files_router, prefix="/api")


@app.get("/")
def read_root():
    return {
        "message": "Welcome to Enterprise Knowledge Assistant API",
        "status": "running"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/db-check")
def db_check(db: Session = Depends(get_db)):
    try:
        # Run a simple query to verify database connection
        result = db.execute(text("SELECT 1")).fetchone()
        if result and result[0] == 1:
            return {
                "database": "connected",
                "message": "Database connection verified successfully"
            }
        else:
            raise HTTPException(
                status_code=500, 
                detail="Database query returned unexpected result"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database connection failed: {str(e)}"
        )
