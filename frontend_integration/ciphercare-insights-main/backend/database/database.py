"""
Database connection and session management.
"""

from sqlmodel import SQLModel, create_engine, Session
from pathlib import Path

# Import all models to register them with SQLModel
from backend.database.models import HospitalData, ModelMetadata, TrainingHistory
from backend.hospitals.models import HospitalData as HospitalDataNew, ModelMetadata as ModelMetadataNew

# Database file path
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATABASE_URL = f"sqlite:///{PROJECT_ROOT / 'ciphercare.db'}"

# Create engine
engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})


def create_db_and_tables():
    """Create database tables."""
    SQLModel.metadata.create_all(engine)


def get_session():
    """Get database session."""
    with Session(engine) as session:
        yield session

