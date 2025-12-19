"""
Hospital Routes

Endpoints for hospital data upload, model retraining, prediction, and metadata.
"""

import csv
import io
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from backend.database.database import get_session
from backend.hospitals.models import HospitalData, ModelMetadata
from backend.ml.train import (
    preprocess_data,
    train_model,
    compute_auroc,
    compute_drift,
    save_model,
    load_model
)

router = APIRouter()

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


class UploadRequest(BaseModel):
    """JSON upload request."""
    data: List[Dict[str, Any]]
    label: Optional[int] = None
    source: str = "upload"


class UploadResponse(BaseModel):
    """Upload response."""
    message: str
    hospital_id: str
    records_inserted: int


class RetrainResponse(BaseModel):
    """Retrain response."""
    message: str
    hospital_id: str
    local_auroc: float
    samples: int
    drift_score: float
    improvement: Optional[float] = None


class PredictRequest(BaseModel):
    """Prediction request."""
    data: Dict[str, Any]


class PredictResponse(BaseModel):
    """Prediction response."""
    risk_score: float
    severity: str
    explanation: Dict[str, Any]


@router.post("/hospitals/{id}/upload", response_model=UploadResponse)
async def upload_data(
    id: str,
    file: Optional[UploadFile] = File(None),
    json_data: Optional[str] = Form(None),
    request: Optional[UploadRequest] = None,
    session: Session = Depends(get_session)
):
    """Upload hospital data via CSV, JSON form, or JSON body."""
    records_inserted = 0
    
    try:
        # Handle JSON body
        if request:
            for record in request.data:
                hospital_data = HospitalData(
                    hospital_id=id,
                    raw_input=record,
                    label=request.label,
                    source=request.source
                )
                session.add(hospital_data)
                records_inserted += 1
        
        # Handle CSV file
        elif file and file.filename:
            if not file.filename.endswith('.csv'):
                raise HTTPException(status_code=400, detail="Only CSV files supported")
            
            content = await file.read()
            csv_content = content.decode('utf-8')
            csv_reader = csv.DictReader(io.StringIO(csv_content))
            
            for row in csv_reader:
                processed_row = {}
                for key, value in row.items():
                    try:
                        if '.' in value:
                            processed_row[key] = float(value)
                        else:
                            processed_row[key] = int(value)
                    except (ValueError, TypeError):
                        processed_row[key] = value
                
                hospital_data = HospitalData(
                    hospital_id=id,
                    raw_input=processed_row,
                    source="upload"
                )
                session.add(hospital_data)
                records_inserted += 1
        
        # Handle JSON form data
        elif json_data:
            try:
                data = json.loads(json_data)
                if isinstance(data, list):
                    records = data
                else:
                    records = [data]
                
                for record in records:
                    hospital_data = HospitalData(
                        hospital_id=id,
                        raw_input=record,
                        source="upload"
                    )
                    session.add(hospital_data)
                    records_inserted += 1
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON")
        
        else:
            raise HTTPException(status_code=400, detail="No data provided")
        
        session.commit()
        
        return UploadResponse(
            message="Data uploaded successfully",
            hospital_id=id,
            records_inserted=records_inserted
        )
    
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hospitals/{id}/retrain", response_model=RetrainResponse)
async def retrain(
    id: str,
    session: Session = Depends(get_session)
):
    """Retrain model for a hospital."""
    try:
        # Load all data for hospital
        statement = select(HospitalData).where(HospitalData.hospital_id == id)
        records = session.exec(statement).all()
        
        if len(records) < 10:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data: {len(records)} records. Need at least 10."
            )
        
        # Convert to format for training
        data_list = []
        labels = []
        for record in records:
            data_list.append(record.raw_input)
            labels.append(record.label if record.label is not None else 0)
        
        # Preprocess
        X, y = preprocess_data(data_list, labels)
        
        # Train
        model = train_model(X, y)
        
        # Compute metrics
        auroc = compute_auroc(model, X, y)
        
        # Get previous metadata for drift calculation
        prev_metadata = session.exec(
            select(ModelMetadata).where(ModelMetadata.hospital_id == id)
        ).first()
        
        drift_score = 0.0
        improvement = None
        if prev_metadata:
            drift_score = compute_drift(auroc, prev_metadata.local_auroc)
            improvement = auroc - prev_metadata.local_auroc
        
        # Save model with feature names
        feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
        model_path = save_model(id, model, feature_names)
        
        # Update or create metadata
        if prev_metadata:
            prev_metadata.local_auroc = auroc
            prev_metadata.samples = len(records)
            prev_metadata.drift_score = drift_score
            prev_metadata.last_trained_at = datetime.utcnow()
            prev_metadata.model_path = model_path
            session.add(prev_metadata)
        else:
            metadata = ModelMetadata(
                hospital_id=id,
                local_auroc=auroc,
                samples=len(records),
                drift_score=drift_score,
                last_trained_at=datetime.utcnow(),
                model_path=model_path
            )
            session.add(metadata)
        
        session.commit()
        
        return RetrainResponse(
            message="Model retrained successfully",
            hospital_id=id,
            local_auroc=round(auroc, 4),
            samples=len(records),
            drift_score=round(drift_score, 4),
            improvement=round(improvement, 4) if improvement is not None else None
        )
    
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")


@router.post("/hospitals/{id}/predict", response_model=PredictResponse)
async def predict(
    id: str,
    request: PredictRequest,
    session: Session = Depends(get_session)
):
    """Run prediction for hospital use case."""
    try:
        # Load model
        model, feature_names = load_model(id)
        
        # Prepare input data
        input_data = request.data
        
        # Convert to feature vector (simplified - should match training format)
        defaults = {
            'age': 65.0, 'gender': 0, 'ethnicity': 0,
            'heart_rate': 72.0, 'systolic_bp': 120.0, 'diastolic_bp': 80.0,
            'resp_rate': 16.0, 'spo2': 98.0, 'temperature': 36.6,
            'glucose': 100.0, 'creatinine': 1.0, 'wbc': 7.0,
            'hemoglobin': 14.0, 'hypertension': 0, 'diabetes': 0
        }
        
        # Map input to features
        mapping = {
            'heartRate': 'heart_rate', 'heart_rate': 'heart_rate',
            'systolicBP': 'systolic_bp', 'systolic_bp': 'systolic_bp',
            'diastolicBP': 'diastolic_bp', 'diastolic_bp': 'diastolic_bp',
            'respRate': 'resp_rate', 'resp_rate': 'resp_rate',
            'SpO2': 'spo2', 'spo2': 'spo2',
            'temp': 'temperature', 'temperature': 'temperature',
        }
        
        for key, value in input_data.items():
            feature_key = mapping.get(key, key)
            if feature_key in defaults:
                try:
                    defaults[feature_key] = float(value) if value is not None else defaults[feature_key]
                except (ValueError, TypeError):
                    pass
            elif key in defaults:
                try:
                    defaults[key] = float(value) if value is not None else defaults[key]
                except (ValueError, TypeError):
                    pass
        
        # Create feature vector
        import pandas as pd
        
        feature_vector = [defaults.get(name, 0.0) for name in feature_names]
        X = pd.DataFrame([feature_vector], columns=feature_names)
        
        # Predict
        if hasattr(model, 'predict_proba'):
            risk_score = float(model.predict_proba(X)[0, 1])
        else:
            risk_score = float(model.predict(X)[0])
            if risk_score < 0:
                risk_score = 0.0
            elif risk_score > 1:
                risk_score = 1.0
        
        # Determine severity
        if risk_score < 0.3:
            severity = "low"
        elif risk_score < 0.7:
            severity = "moderate"
        else:
            severity = "high"
        
        # Compute explanation (feature importance)
        explanation = {}
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            for i, name in enumerate(feature_names):
                if i < len(importances):
                    explanation[name] = float(importances[i])
        else:
            # Default explanation
            explanation = {k: abs(v - defaults.get(k, 0)) * 0.1 for k, v in input_data.items()}
        
        return PredictResponse(
            risk_score=round(risk_score, 4),
            severity=severity,
            explanation=explanation
        )
    
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Model not found for hospital {id}. Please retrain first."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.get("/hospitals/{id}/metadata")
async def get_metadata(
    id: str,
    session: Session = Depends(get_session)
):
    """Get model metadata for a hospital."""
    metadata = session.exec(
        select(ModelMetadata).where(ModelMetadata.hospital_id == id)
    ).first()
    
    if not metadata:
        raise HTTPException(status_code=404, detail=f"No metadata found for hospital {id}")
    
    return {
        "hospital_id": metadata.hospital_id,
        "local_auroc": metadata.local_auroc,
        "samples": metadata.samples,
        "drift_score": metadata.drift_score,
        "last_trained_at": metadata.last_trained_at.isoformat(),
        "model_path": metadata.model_path
    }

