"""
FastAPI Backend for CipherCare IoMT Deterioration Risk Prediction

Main application entry point.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.routers import prediction, hospital_data, hospital_inference, retrain, model_metadata
from backend.hospitals import routes
from backend.database.database import create_db_and_tables


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    create_db_and_tables()
    yield
    # Shutdown (if needed)


app = FastAPI(
    title="CipherCare ML API",
    description="Deterioration risk prediction API for IoMT monitoring",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend integration - must be added before routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Include routers
app.include_router(prediction.router, prefix="/api", tags=["prediction"])
app.include_router(hospital_data.router, prefix="/api", tags=["hospital-data"])
app.include_router(hospital_inference.router, prefix="/api", tags=["hospital-inference"])
app.include_router(retrain.router, prefix="/api", tags=["retraining"])
app.include_router(model_metadata.router, prefix="/api", tags=["model-metadata"])
app.include_router(routes.router, prefix="/api", tags=["hospitals"])

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "CipherCare ML API",
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

