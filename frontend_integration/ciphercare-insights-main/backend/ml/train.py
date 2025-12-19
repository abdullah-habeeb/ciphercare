"""
ML Training Functions

Simple sklearn-based training pipeline for hospital models.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import joblib

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def preprocess_data(
    data_list: List[Dict[str, Any]],
    labels: List[int]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess raw data into feature matrix.
    
    Args:
        data_list: List of dictionaries with raw input data
        labels: List of labels (0 or 1)
    
    Returns:
        X: Feature DataFrame
        y: Label Series
    """
    # Default feature set
    defaults = {
        'age': 65.0, 'gender': 0, 'ethnicity': 0,
        'heart_rate': 72.0, 'systolic_bp': 120.0, 'diastolic_bp': 80.0,
        'resp_rate': 16.0, 'spo2': 98.0, 'temperature': 36.6,
        'glucose': 100.0, 'creatinine': 1.0, 'wbc': 7.0,
        'hemoglobin': 14.0, 'hypertension': 0, 'diabetes': 0
    }
    
    # Feature order
    feature_order = [
        'age', 'gender', 'ethnicity',
        'heart_rate', 'systolic_bp', 'diastolic_bp', 'resp_rate',
        'spo2', 'temperature',
        'glucose', 'creatinine', 'wbc', 'hemoglobin',
        'hypertension', 'diabetes'
    ]
    
    # Convert data to feature vectors
    feature_vectors = []
    for record in data_list:
        features = {}
        for key in feature_order:
            # Try various key formats
            value = None
            if key in record:
                value = record[key]
            elif key.replace('_', '') in record:
                value = record[key.replace('_', '')]
            else:
                # Try camelCase
                camel_key = ''.join(word.capitalize() if i > 0 else word 
                                   for i, word in enumerate(key.split('_')))
                if camel_key in record:
                    value = record[camel_key]
                elif camel_key.lower() in record:
                    value = record[camel_key.lower()]
            
            if value is not None:
                try:
                    features[key] = float(value)
                except (ValueError, TypeError):
                    features[key] = defaults.get(key, 0.0)
            else:
                features[key] = defaults.get(key, 0.0)
        
        feature_vectors.append([features[k] for k in feature_order])
    
    X = pd.DataFrame(feature_vectors, columns=feature_order)
    y = pd.Series(labels)
    
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series):
    """
    Train a RandomForest classifier.
    
    Args:
        X: Feature matrix
        y: Labels
    
    Returns:
        Trained model
    """
    # Use RandomForest for simplicity and robustness
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'  # Handle imbalance
    )
    
    model.fit(X, y)
    
    return model


def compute_auroc(model, X: pd.DataFrame, y: pd.Series) -> float:
    """
    Compute AUROC score.
    
    Args:
        model: Trained model
        X: Features
        y: True labels
    
    Returns:
        AUROC score
    """
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X)[:, 1]
    else:
        y_pred_proba = model.predict(X)
        # Normalize to [0, 1]
        y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min() + 1e-8)
    
    if len(y.unique()) < 2:
        return 0.5  # Default if only one class
    
    try:
        auroc = roc_auc_score(y, y_pred_proba)
        return max(0.5, min(1.0, auroc))  # Clamp between 0.5 and 1.0
    except ValueError:
        return 0.5


def compute_drift(current_auroc: float, previous_auroc: float) -> float:
    """
    Compute drift score as absolute difference.
    
    Args:
        current_auroc: Current model AUROC
        previous_auroc: Previous model AUROC
    
    Returns:
        Drift score
    """
    return abs(current_auroc - previous_auroc)


def save_model(
    hospital_id: str,
    model,
    feature_names: Optional[List[str]] = None
) -> str:
    """
    Save model to disk.
    
    Args:
        hospital_id: Hospital identifier
        model: Trained model
        feature_names: List of feature names
    
    Returns:
        Path to saved model
    """
    model_path = MODELS_DIR / f"{hospital_id}.pkl"
    
    # Save model with metadata
    model_data = {
        'model': model,
        'feature_names': feature_names
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    return str(model_path)


def load_model(hospital_id: str) -> Tuple[Any, List[str]]:
    """
    Load model from disk.
    
    Args:
        hospital_id: Hospital identifier
    
    Returns:
        model: Loaded model
        feature_names: List of feature names
    """
    model_path = MODELS_DIR / f"{hospital_id}.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    feature_names = model_data.get('feature_names', [
        'age', 'gender', 'ethnicity',
        'heart_rate', 'systolic_bp', 'diastolic_bp', 'resp_rate',
        'spo2', 'temperature',
        'glucose', 'creatinine', 'wbc', 'hemoglobin',
        'hypertension', 'diabetes'
    ])
    
    return model, feature_names

