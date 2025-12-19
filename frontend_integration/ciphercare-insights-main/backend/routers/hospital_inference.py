"""
Hospital-Specific Inference Router

Handles per-hospital prediction endpoints with use-case specific interpretations.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.ml.retrain import load_hospital_model

router = APIRouter()

PROJECT_ROOT = Path(__file__).parent.parent.parent


class HospitalPredictionRequest(BaseModel):
    """Hospital-specific prediction request."""
    data: Dict[str, Any] = Field(..., description="Hospital-specific input data")


class HospitalPredictionResponse(BaseModel):
    """Hospital-specific prediction response."""
    severity: str = Field(..., description="Severity category")
    risk_score: float = Field(..., ge=0, le=1, description="Risk probability")
    explanation: Dict[str, str] = Field(..., description="Feature contributions")
    recommended_action: str = Field(..., description="Clinical recommendation")


def infer_severity_from_risk(risk_score: float) -> tuple[str, str]:
    """Infer severity and action from risk score."""
    if risk_score < 0.3:
        return "stable", "Continue monitoring"
    elif risk_score < 0.7:
        return "moderate", "Re-evaluate vitals in 5 min"
    else:
        return "critical", "Notify attending physician immediately"


def prepare_features_for_hospital(
    data: Dict[str, Any],
    hospital_id: str,
    feature_names: list
) -> pd.DataFrame:
    """Prepare feature vector from hospital-specific input."""
    # Default values
    defaults = {
        'age': 65.0,
        'gender': 0,
        'ethnicity': 0,
        'heart_rate': 72.0,
        'systolic_bp': 120.0,
        'diastolic_bp': 80.0,
        'resp_rate': 16.0,
        'spo2': 98.0,
        'temperature': 36.6,
        'glucose': 100.0,
        'creatinine': 1.0,
        'wbc': 7.0,
        'hemoglobin': 14.0,
        'hypertension': 0,
        'diabetes': 0
    }
    
    # Map input data to features (handle various naming conventions)
    mapping = {
        'heartRate': 'heart_rate',
        'heart_rate': 'heart_rate',
        'systolicBP': 'systolic_bp',
        'systolic_bp': 'systolic_bp',
        'diastolicBP': 'diastolic_bp',
        'diastolic_bp': 'diastolic_bp',
        'respRate': 'resp_rate',
        'resp_rate': 'resp_rate',
        'SpO2': 'spo2',
        'spo2': 'spo2',
        'temp': 'temperature',
        'temperature': 'temperature',
    }
    
    # Update defaults with provided data
    for key, value in data.items():
        feature_key = mapping.get(key, key)
        if feature_key in defaults:
            defaults[feature_key] = float(value) if value is not None else defaults[feature_key]
        elif key in defaults:
            defaults[key] = float(value) if value is not None else defaults[key]
    
    # Create feature vector in correct order
    features = {name: defaults.get(name, 0.0) for name in feature_names}
    
    return pd.DataFrame([features])


def compute_explanation(
    model,
    X_scaled: pd.DataFrame,
    feature_names: list,
    input_data: Dict[str, Any],
    risk_score: float
) -> Dict[str, str]:
    """Compute feature contributions for explanation."""
    importance = model.feature_importances_
    
    # Map input keys to feature names
    input_to_feature = {
        'heartRate': 'heart_rate',
        'heart_rate': 'heart_rate',
        'spo2': 'spo2',
        'SpO2': 'spo2',
        'temperature': 'temperature',
        'temp': 'temperature',
        'respRate': 'resp_rate',
        'resp_rate': 'resp_rate',
        'systolicBP': 'systolic_bp',
        'systolic_bp': 'systolic_bp',
        'diastolicBP': 'diastolic_bp',
        'diastolic_bp': 'diastolic_bp',
    }
    
    explanations = {}
    
    # Compute contributions for each input feature
    for input_key, value in input_data.items():
        if value is None:
            continue
        
        feature_name = input_to_feature.get(input_key, input_key)
        if feature_name in feature_names:
            idx = feature_names.index(feature_name)
            feature_importance = importance[idx]
            feature_value = X_scaled.iloc[0, idx]
            
            # Contribution as percentage change
            contribution = feature_importance * abs(feature_value) * 0.1
            
            # Format as percentage change
            if contribution > 0.01:
                explanations[input_key] = f"+{contribution * 100:.1f}%"
            elif contribution < -0.01:
                explanations[input_key] = f"{contribution * 100:.1f}%"
            else:
                explanations[input_key] = "0.0%"
    
    # If no explanations, provide default
    if not explanations:
        explanations = {
            "risk_score": f"{risk_score:.2f}",
            "model_confidence": "High"
        }
    
    return explanations


@router.post("/hospitals/{hospital_id}/predict", response_model=HospitalPredictionResponse)
async def predict_hospital_use_case(
    hospital_id: str,
    request: HospitalPredictionRequest
):
    """
    Run inference for hospital-specific use case.
    
    Each hospital has different input requirements:
    - Hospital A: ECG conditions
    - Hospital B: Vitals deterioration
    - Hospital C: X-ray metadata
    - Hospital D: Geriatric ECG
    - Hospital E: Multi-modal inputs
    """
    try:
        # Load hospital-specific model
        model, scaler, feature_names = load_hospital_model(hospital_id)
        
        # Prepare features
        X = prepare_features_for_hospital(request.data, hospital_id, feature_names)
        
        # Scale features
        X_scaled = scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
        
        # Predict
        risk_score = float(model.predict_proba(X_scaled_df)[0, 1])
        
        # Infer severity and action
        severity, action = infer_severity_from_risk(risk_score)
        
        # Compute explanation
        explanation = compute_explanation(
            model, X_scaled_df, feature_names, request.data, risk_score
        )
        
        return HospitalPredictionResponse(
            severity=severity,
            risk_score=round(risk_score, 4),
            explanation=explanation,
            recommended_action=action
        )
    
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Model not found for hospital {hospital_id}. Please retrain the model first."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

