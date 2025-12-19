"""
Database Models for CipherCare Hospital Data and Model Metadata

Uses SQLModel for type-safe database operations.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlmodel import SQLModel, Field, JSON, Column
from sqlalchemy import DateTime
from sqlalchemy.sql import func


class HospitalDataBase(SQLModel):
    """Base model for hospital data."""
    hospital_id: str = Field(index=True)
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
    raw_data: Dict[str, Any] = Field(sa_column=Column(JSON))
    label: Optional[int] = Field(default=None)
    source: str = Field(default="upload")  # "upload", "batch", "synthetic"


class HospitalData(HospitalDataBase, table=True):
    """Hospital data table for storing patient vitals and training data."""
    __tablename__ = "hospital_data"
    
    id: Optional[int] = Field(default=None, primary_key=True)


class ModelMetadataBase(SQLModel):
    """Base model for model metadata."""
    hospital_id: str = Field(index=True)
    model_type: str
    local_auroc: float
    global_auroc: Optional[float] = None
    last_trained_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
    training_samples: int
    drift_score: Optional[float] = None
    notes: Optional[str] = None


class ModelMetadata(ModelMetadataBase, table=True):
    """Model metadata table for tracking model performance and training history."""
    __tablename__ = "model_metadata"
    
    id: Optional[int] = Field(default=None, primary_key=True)


class TrainingHistoryBase(SQLModel):
    """Base model for training history."""
    hospital_id: str = Field(index=True)
    training_date: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
    local_auroc: float
    global_auroc: Optional[float] = None
    samples_used: int
    improvement: Optional[float] = None
    training_duration_seconds: Optional[float] = None
    notes: Optional[str] = None


class TrainingHistory(TrainingHistoryBase, table=True):
    """Training history table for tracking model retraining events."""
    __tablename__ = "training_history"
    
    id: Optional[int] = Field(default=None, primary_key=True)

