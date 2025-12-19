"""
Prediction Router for Deterioration Risk API

Handles real-time vitals prediction requests from IoMT frontend.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional

router = APIRouter()

# Load model and preprocessing pipeline
# Get project root (2 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "deterioration_model.pkl"
PIPELINE_PATH = PROJECT_ROOT / "models" / "preprocessing_pipeline.pkl"

# Global variables for loaded models
_model = None
_pipeline = None
_feature_names = None

def load_model():
    """Lazy load model and pipeline."""
    global _model, _pipeline, _feature_names
    
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please run train_model.py first.")
        if not PIPELINE_PATH.exists():
            raise FileNotFoundError(f"Pipeline not found at {PIPELINE_PATH}. Please run train_model.py first.")
        
        print("Loading model and pipeline...")
        with open(MODEL_PATH, 'rb') as f:
            _model = pickle.load(f)
        
        with open(PIPELINE_PATH, 'rb') as f:
            pipeline_data = pickle.load(f)
            _pipeline = pipeline_data['scaler']
            _feature_names = pipeline_data['feature_names']
        
        print("✅ Model and pipeline loaded successfully")
    
    return _model, _pipeline, _feature_names

# Request/Response models
class VitalsRequest(BaseModel):
    """Vitals data from IoMT frontend."""
    heartRate: float = Field(..., ge=0, le=250, description="Heart rate in BPM")
    spo2: float = Field(..., ge=0, le=100, description="Oxygen saturation percentage")
    temperature: float = Field(..., ge=30, le=45, description="Body temperature in Celsius")
    respRate: Optional[float] = Field(None, ge=0, le=60, description="Respiratory rate per minute")
    systolicBP: Optional[float] = Field(None, ge=50, le=250, description="Systolic blood pressure")
    diastolicBP: Optional[float] = Field(None, ge=30, le=150, description="Diastolic blood pressure")
    
    class Config:
        json_schema_extra = {
            "example": {
                "heartRate": 85,
                "spo2": 97,
                "temperature": 37.1,
                "respRate": 18,
                "systolicBP": 120,
                "diastolicBP": 80
            }
        }

class PredictionResponse(BaseModel):
    """Prediction response with risk score and recommendations."""
    riskScore: float = Field(..., ge=0, le=1, description="Deterioration risk probability")
    severity: str = Field(..., description="Severity category: Stable, Warning, or Critical")
    recommendedAction: str = Field(..., description="Clinical action recommendation")
    featureContributions: Dict[str, float] = Field(..., description="Feature contribution scores")

def infer_severity(risk_score: float) -> tuple[str, str]:
    """
    Infer severity category and recommended action from risk score.
    
    Args:
        risk_score: Probability of deterioration (0-1)
        
    Returns:
        (severity, action): Tuple of severity category and recommended action
    """
    if risk_score < 0.3:
        return "Stable", "Continue monitoring"
    elif risk_score < 0.7:
        return "Warning", "Reassess vitals & run labs"
    else:
        return "Critical", "Notify attending physician immediately"

def compute_feature_contributions(
    model, 
    X_scaled: pd.DataFrame, 
    feature_names: list,
    vitals: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute SHAP-like feature contributions using model's feature importance.
    
    Args:
        model: Trained XGBoost model
        X_scaled: Scaled feature vector
        feature_names: List of feature names
        vitals: Original vitals dictionary
        
    Returns:
        Dictionary of feature contributions
    """
    # Get feature importance
    importance = model.feature_importances_
    
    # Map vitals to feature contributions
    contributions = {}
    
    # Direct mappings
    vitals_to_features = {
        'heartRate': 'heart_rate',
        'spo2': 'spo2',
        'temperature': 'temperature',
        'respRate': 'resp_rate',
        'systolicBP': 'systolic_bp',
        'diastolicBP': 'diastolic_bp'
    }
    
    for vital_key, feature_name in vitals_to_features.items():
        if vital_key in vitals and vitals[vital_key] is not None:
            # Find feature index
            if feature_name in feature_names:
                idx = feature_names.index(feature_name)
                # Contribution is importance * normalized feature value
                feature_value = X_scaled.iloc[0, idx]
                contributions[vital_key] = round(importance[idx] * abs(feature_value) * 0.1, 3)
            else:
                contributions[vital_key] = 0.0
        else:
            contributions[vital_key] = 0.0
    
    # Normalize contributions to sum to risk score
    total_contribution = sum(abs(v) for v in contributions.values())
    if total_contribution > 0:
        scale_factor = 1.0 / total_contribution if total_contribution > 0 else 1.0
        contributions = {k: round(v * scale_factor * 0.5, 3) for k, v in contributions.items()}
    
    return contributions

def prepare_features(vitals: VitalsRequest) -> pd.DataFrame:
    """
    Prepare feature vector from vitals, filling missing values with defaults.
    
    Args:
        vitals: Vitals request object
        
    Returns:
        DataFrame with all required features
    """
    # Default values for missing features (based on normal ranges)
    defaults = {
        'age': 65.0,  # Average age
        'gender': 0,  # Female
        'ethnicity': 0,  # White
        'heart_rate': vitals.heartRate,
        'systolic_bp': vitals.systolicBP if vitals.systolicBP else 120.0,
        'diastolic_bp': vitals.diastolicBP if vitals.diastolicBP else 80.0,
        'resp_rate': vitals.respRate if vitals.respRate else 16.0,
        'spo2': vitals.spo2,
        'temperature': vitals.temperature,
        'glucose': 100.0,  # Normal
        'creatinine': 1.0,  # Normal
        'wbc': 7.0,  # Normal
        'hemoglobin': 14.0,  # Normal
        'hypertension': 0,
        'diabetes': 0
    }
    
    # Create feature vector in correct order
    feature_order = [
        'age', 'gender', 'ethnicity',
        'heart_rate', 'systolic_bp', 'diastolic_bp', 'resp_rate',
        'spo2', 'temperature',
        'glucose', 'creatinine', 'wbc', 'hemoglobin',
        'hypertension', 'diabetes'
    ]
    
    features = {name: defaults.get(name, 0.0) for name in feature_order}
    
    return pd.DataFrame([features])

@router.post("/predict", response_model=PredictionResponse)
async def predict_deterioration(vitals: VitalsRequest):
    """
    Predict deterioration risk from IoMT vitals.
    
    Args:
        vitals: Vitals data from frontend
        
    Returns:
        Prediction response with risk score, severity, and recommendations
    """
    try:
        # Load model (lazy loading)
        model, scaler, feature_names = load_model()
        
        # Prepare features
        X = prepare_features(vitals)
        
        # Scale features
        X_scaled = scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
        
        # Predict
        risk_score = float(model.predict_proba(X_scaled_df)[0, 1])
        
        # Infer severity and action
        severity, action = infer_severity(risk_score)
        
        # Compute feature contributions
        vitals_dict = vitals.model_dump()
        contributions = compute_feature_contributions(
            model, X_scaled_df, feature_names, vitals_dict
        )
        
        return PredictionResponse(
            riskScore=round(risk_score, 4),
            severity=severity,
            recommendedAction=action,
            featureContributions=contributions
        )
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not available: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.get("/model-info")
async def get_model_info():
    """Get information about the loaded model."""
    try:
        model, _, feature_names = load_model()
        return {
            "status": "loaded",
            "model_type": type(model).__name__,
            "n_features": len(feature_names),
            "features": feature_names
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

