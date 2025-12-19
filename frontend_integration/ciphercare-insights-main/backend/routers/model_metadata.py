"""
Model Metadata Router

Handles retrieval of model metadata and training history.
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlmodel import Session, select
from typing import List, Optional

from backend.database.database import get_session
from backend.database.models import ModelMetadata, TrainingHistory

router = APIRouter()


@router.get("/hospitals/{hospital_id}/model-metadata")
async def get_model_metadata(
    hospital_id: str,
    session: Session = Depends(get_session)
):
    """Get current model metadata for a hospital."""
    statement = select(ModelMetadata).where(ModelMetadata.hospital_id == hospital_id)
    metadata = session.exec(statement).first()
    
    if not metadata:
        raise HTTPException(status_code=404, detail=f"No model metadata found for hospital {hospital_id}")
    
    return {
        "id": metadata.id,
        "hospital_id": metadata.hospital_id,
        "model_type": metadata.model_type,
        "local_auroc": metadata.local_auroc,
        "global_auroc": metadata.global_auroc,
        "last_trained_at": metadata.last_trained_at.isoformat() if metadata.last_trained_at else None,
        "training_samples": metadata.training_samples,
        "drift_score": metadata.drift_score,
        "notes": metadata.notes
    }


@router.get("/hospitals/{hospital_id}/training-history")
async def get_training_history(
    hospital_id: str,
    limit: int = 20,
    session: Session = Depends(get_session)
):
    """Get training history for a hospital."""
    statement = (
        select(TrainingHistory)
        .where(TrainingHistory.hospital_id == hospital_id)
        .order_by(TrainingHistory.training_date.desc())
        .limit(limit)
    )
    history = session.exec(statement).all()
    
    return {
        "hospital_id": hospital_id,
        "count": len(history),
        "history": [
            {
                "id": h.id,
                "training_date": h.training_date.isoformat(),
                "local_auroc": h.local_auroc,
                "global_auroc": h.global_auroc,
                "samples_used": h.samples_used,
                "improvement": h.improvement,
                "training_duration_seconds": h.training_duration_seconds,
                "notes": h.notes
            }
            for h in history
        ]
    }


@router.get("/hospitals/{hospital_id}/metadata/summary")
async def get_metadata_summary(
    hospital_id: str,
    session: Session = Depends(get_session)
):
    """Get summary of model metadata and recent training."""
    # Get current metadata
    metadata_stmt = select(ModelMetadata).where(ModelMetadata.hospital_id == hospital_id)
    metadata = session.exec(metadata_stmt).first()
    
    # Get recent training history
    history_stmt = (
        select(TrainingHistory)
        .where(TrainingHistory.hospital_id == hospital_id)
        .order_by(TrainingHistory.training_date.desc())
        .limit(5)
    )
    recent_history = session.exec(history_stmt).all()
    
    return {
        "hospital_id": hospital_id,
        "current_metadata": {
            "local_auroc": metadata.local_auroc if metadata else None,
            "training_samples": metadata.training_samples if metadata else None,
            "last_trained_at": metadata.last_trained_at.isoformat() if metadata and metadata.last_trained_at else None,
            "drift_score": metadata.drift_score if metadata else None,
            "model_type": metadata.model_type if metadata else None,
        },
        "recent_training": [
            {
                "training_date": h.training_date.isoformat(),
                "local_auroc": h.local_auroc,
                "samples_used": h.samples_used,
                "improvement": h.improvement
            }
            for h in recent_history
        ]
    }

