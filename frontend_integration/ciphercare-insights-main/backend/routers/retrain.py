"""
Model Retraining Router

Handles model retraining requests for hospitals.
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlmodel import Session
from typing import Dict, Any

from backend.database.database import get_session
from backend.ml.retrain import retrain_hospital_model

router = APIRouter()


@router.post("/hospitals/{hospital_id}/retrain")
async def retrain_model(
    hospital_id: str,
    model_type: str = "XGBoost",
    session: Session = Depends(get_session)
) -> Dict[str, Any]:
    """
    Retrain model for a specific hospital.
    
    Can be triggered:
    - Manually from UI
    - Automatically when new data is uploaded
    """
    try:
        result = retrain_hospital_model(session, hospital_id, model_type)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")

