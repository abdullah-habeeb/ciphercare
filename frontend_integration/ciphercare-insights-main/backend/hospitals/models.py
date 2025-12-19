"""
Hospital Data and Model Metadata Models

SQLModel tables for storing hospital data and model metadata.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlmodel import SQLModel, Field, JSON, Column
from sqlalchemy import DateTime
from sqlalchemy.sql import func


class HospitalData(SQLModel, table=True):
    """Hospital data table for storing uploaded patient data."""
    __tablename__ = "hospital_data_new"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    hospital_id: str = Field(index=True)
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
    raw_input: Dict[str, Any] = Field(sa_column=Column(JSON))
    label: Optional[int] = Field(default=None)
    source: str = Field(default="upload")  # "upload", "batch", "synthetic"


class ModelMetadata(SQLModel, table=True):
    """Model metadata table for tracking model performance."""
    __tablename__ = "model_metadata_new"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    hospital_id: str = Field(index=True, unique=True)
    local_auroc: float
    samples: int
    drift_score: float = Field(default=0.0)
    last_trained_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
    model_path: str

