"""
Model Retraining Engine for Hospital-Specific Models

Handles retraining of models per hospital with data from database.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from sqlmodel import Session, select

from backend.database.models import HospitalData, ModelMetadata, TrainingHistory
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import xgboost as xgb

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_hospital_data(session: Session, hospital_id: str) -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    Load all data for a hospital from database.
    
    Returns:
        X: Features DataFrame
        y: Target Series
        feature_names: List of feature names
    """
    statement = select(HospitalData).where(HospitalData.hospital_id == hospital_id)
    records = session.exec(statement).all()
    
    if len(records) == 0:
        raise ValueError(f"No data found for hospital {hospital_id}")
    
    # Convert records to DataFrame
    data_list = []
    labels = []
    
    for record in records:
        raw_data = record.raw_data
        # Extract features from raw_data
        # Default feature set (same as train_model.py)
        feature_dict = {
            'age': raw_data.get('age', 65.0),
            'gender': raw_data.get('gender', 0),
            'ethnicity': raw_data.get('ethnicity', 0),
            'heart_rate': raw_data.get('heart_rate', raw_data.get('heartRate', 72.0)),
            'systolic_bp': raw_data.get('systolic_bp', raw_data.get('systolicBP', 120.0)),
            'diastolic_bp': raw_data.get('diastolic_bp', raw_data.get('diastolicBP', 80.0)),
            'resp_rate': raw_data.get('resp_rate', raw_data.get('respRate', 16.0)),
            'spo2': raw_data.get('spo2', raw_data.get('SpO2', 98.0)),
            'temperature': raw_data.get('temperature', raw_data.get('temp', 36.6)),
            'glucose': raw_data.get('glucose', 100.0),
            'creatinine': raw_data.get('creatinine', 1.0),
            'wbc': raw_data.get('wbc', 7.0),
            'hemoglobin': raw_data.get('hemoglobin', 14.0),
            'hypertension': raw_data.get('hypertension', 0),
            'diabetes': raw_data.get('diabetes', 0),
        }
        data_list.append(feature_dict)
        
        # Use label if available, otherwise try to infer from raw_data
        if record.label is not None:
            labels.append(record.label)
        elif 'deterioration_risk' in raw_data:
            labels.append(int(raw_data['deterioration_risk']))
        else:
            # Default to 0 if no label
            labels.append(0)
    
    X = pd.DataFrame(data_list)
    y = pd.Series(labels)
    
    feature_names = list(X.columns)
    
    return X, y, feature_names


def handle_imbalance(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Handle class imbalance using SMOTE or class weights."""
    if SMOTE_AVAILABLE and len(y_train.unique()) == 2:
        try:
            smote = SMOTE(random_state=42, k_neighbors=min(3, len(y_train[y_train==1]) - 1))
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled)
        except Exception:
            return X_train, y_train
    return X_train, y_train


def train_hospital_model(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list,
    hospital_id: str
) -> Tuple[xgb.XGBClassifier, StandardScaler, Dict[str, float]]:
    """
    Train a model for a specific hospital.
    
    Returns:
        model: Trained XGBoost model
        scaler: Fitted StandardScaler
        metrics: Dictionary of performance metrics
    """
    # Split data
    if len(X) < 20:
        # Too little data, use all for training
        X_train, X_test = X, X
        y_train, y_test = y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
        )
    
    # Handle imbalance
    X_train_balanced, y_train_balanced = handle_imbalance(X_train, y_train)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    # Train model
    scale_pos_weight = len(y_train_balanced[y_train_balanced == 0]) / max(len(y_train_balanced[y_train_balanced == 1]), 1)
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='auc'
    )
    
    try:
        model.fit(
            X_train_scaled, y_train_balanced,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
    except TypeError:
        model.fit(
            X_train_scaled, y_train_balanced,
            eval_set=[(X_test_scaled, y_test)],
            early_stopping_rounds=20,
            verbose=False
        )
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)
    
    metrics = {
        'local_auroc': roc_auc_score(y_test, y_pred_proba) if len(y_test.unique()) > 1 else 0.5,
        'f1': f1_score(y_test, y_pred) if len(y_test.unique()) > 1 else 0.0,
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
    }
    
    return model, scaler, metrics


def save_hospital_model(
    hospital_id: str,
    model: xgb.XGBClassifier,
    scaler: StandardScaler,
    feature_names: list
):
    """Save model and preprocessing pipeline for a hospital."""
    model_path = MODELS_DIR / f"{hospital_id}_model.pkl"
    pipeline_path = MODELS_DIR / f"{hospital_id}_pipeline.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    pipeline = {
        'scaler': scaler,
        'feature_names': feature_names
    }
    with open(pipeline_path, 'wb') as f:
        pickle.dump(pipeline, f)


def load_hospital_model(hospital_id: str) -> Tuple[xgb.XGBClassifier, StandardScaler, list]:
    """Load model and pipeline for a hospital."""
    model_path = MODELS_DIR / f"{hospital_id}_model.pkl"
    pipeline_path = MODELS_DIR / f"{hospital_id}_pipeline.pkl"
    
    if not model_path.exists() or not pipeline_path.exists():
        raise FileNotFoundError(f"Model not found for hospital {hospital_id}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(pipeline_path, 'rb') as f:
        pipeline = pickle.load(f)
        scaler = pipeline['scaler']
        feature_names = pipeline['feature_names']
    
    return model, scaler, feature_names


def retrain_hospital_model(
    session: Session,
    hospital_id: str,
    model_type: str = "XGBoost"
) -> Dict[str, Any]:
    """
    Retrain model for a hospital using all available data.
    
    Returns:
        Dictionary with training results
    """
    start_time = datetime.utcnow()
    
    # Load data
    X, y, feature_names = load_hospital_data(session, hospital_id)
    
    # Train model
    model, scaler, metrics = train_hospital_model(X, y, feature_names, hospital_id)
    
    # Save model
    save_hospital_model(hospital_id, model, scaler, feature_names)
    
    # Get previous metadata for comparison
    statement = select(ModelMetadata).where(ModelMetadata.hospital_id == hospital_id)
    prev_metadata = session.exec(statement).first()
    
    prev_auroc = prev_metadata.local_auroc if prev_metadata else None
    improvement = (metrics['local_auroc'] - prev_auroc) if prev_auroc else None
    
    # Calculate drift score (simple: difference from previous AUROC)
    drift_score = abs(improvement) if improvement is not None else None
    
    # Update or create metadata
    if prev_metadata:
        prev_metadata.local_auroc = metrics['local_auroc']
        prev_metadata.last_trained_at = datetime.utcnow()
        prev_metadata.training_samples = len(X)
        prev_metadata.drift_score = drift_score
        session.add(prev_metadata)
    else:
        metadata = ModelMetadata(
            hospital_id=hospital_id,
            model_type=model_type,
            local_auroc=metrics['local_auroc'],
            training_samples=len(X),
            drift_score=drift_score
        )
        session.add(metadata)
    
    # Create training history entry
    training_duration = (datetime.utcnow() - start_time).total_seconds()
    history = TrainingHistory(
        hospital_id=hospital_id,
        local_auroc=metrics['local_auroc'],
        samples_used=len(X),
        improvement=improvement,
        training_duration_seconds=training_duration
    )
    session.add(history)
    
    session.commit()
    
    return {
        "message": "Model retrained successfully",
        "hospital_id": hospital_id,
        "local_auroc": round(metrics['local_auroc'], 4),
        "samples_used": len(X),
        "improvement": round(improvement, 4) if improvement is not None else None,
        "f1": round(metrics['f1'], 4),
        "precision": round(metrics['precision'], 4),
        "recall": round(metrics['recall'], 4),
        "training_duration_seconds": round(training_duration, 2)
    }

