"""
Model Training Script for Deterioration Risk Prediction

Trains an XGBoost classifier with automatic imbalance handling.
Saves model, preprocessing pipeline, and feature importance.
"""

import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
import xgboost as xgb
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Warning: imbalanced-learn not available, using class weights instead")

# Set random seed
np.random.seed(42)

# Get project root (parent of ml directory)
PROJECT_ROOT = Path(__file__).parent.parent

# Create models directory
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

def load_and_prepare_data(csv_path=None):
    if csv_path is None:
        csv_path = PROJECT_ROOT / "data" / "patient_vitals.csv"
    """
    Load dataset and prepare features.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        X, y: Features and target
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Feature columns (exclude patient_id and target)
    feature_cols = [
        'age', 'gender', 'ethnicity',
        'heart_rate', 'systolic_bp', 'diastolic_bp', 'resp_rate', 
        'spo2', 'temperature',
        'glucose', 'creatinine', 'wbc', 'hemoglobin',
        'hypertension', 'diabetes'
    ]
    
    X = df[feature_cols].copy()
    y = df['deterioration_risk'].copy()
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"Class imbalance: {len(y[y==0]) / len(y[y==1]):.1f}:1")
    
    return X, y, feature_cols

def handle_imbalance(X_train, y_train, strategy='smote'):
    """
    Handle class imbalance using SMOTE oversampling or class weights.
    
    Args:
        X_train: Training features
        y_train: Training labels
        strategy: 'smote' or 'class_weight'
        
    Returns:
        X_resampled, y_resampled: Balanced training data
    """
    if strategy == 'smote' and SMOTE_AVAILABLE:
        try:
            print("\nApplying SMOTE oversampling...")
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE: {len(y_resampled[y_resampled==0])} stable, {len(y_resampled[y_resampled==1])} high-risk")
            return X_resampled, y_resampled
        except Exception as e:
            print(f"SMOTE failed: {e}. Falling back to class weights.")
            return X_train, y_train
    else:
        print("\nUsing class weights for imbalance handling...")
        return X_train, y_train

def train_model(X_train, y_train, X_val, y_val):
    """
    Train XGBoost classifier with optimal hyperparameters.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        
    Returns:
        Trained model
    """
    print("\nTraining XGBoost classifier...")
    
    # XGBoost with class weights for imbalance
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    
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
    
    # Train with early stopping (new XGBoost API)
    try:
        # New API (XGBoost 2.0+)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    except TypeError:
        # Fallback for older versions
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_test, y_test: Test data
        
    Returns:
        Dictionary of metrics
    """
    print("\nEvaluating model...")
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Metrics
    auroc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"AUROC: {auroc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Stable', 'High-Risk']))
    
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return {
        'auroc': auroc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def get_feature_importance(model, feature_names):
    """
    Extract and sort feature importance.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance
    """
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    return feature_importance

def save_model_and_pipeline(model, scaler, feature_names, feature_importance):
    """
    Save model, scaler, and metadata.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        feature_names: Feature column names
        feature_importance: Feature importance DataFrame
    """
    print("\nSaving model and pipeline...")
    
    # Save model
    model_path = MODELS_DIR / "deterioration_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")
    
    # Save preprocessing pipeline
    pipeline_path = MODELS_DIR / "preprocessing_pipeline.pkl"
    pipeline = {
        'scaler': scaler,
        'feature_names': feature_names
    }
    with open(pipeline_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Preprocessing pipeline saved to {pipeline_path}")
    
    # Save feature importance
    importance_path = MODELS_DIR / "feature_importance.csv"
    feature_importance.to_csv(importance_path, index=False)
    print(f"Feature importance saved to {importance_path}")

def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Deterioration Risk Model Training")
    print("=" * 60)
    
    # Load data
    X, y, feature_names = load_and_prepare_data()
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nData splits:")
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Handle imbalance
    X_train_balanced, y_train_balanced = handle_imbalance(X_train, y_train, strategy='smote')
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for XGBoost
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    # Train model
    model = train_model(X_train_scaled, y_train_balanced, X_val_scaled, y_val)
    
    # Evaluate
    metrics = evaluate_model(model, X_test_scaled, y_test)
    
    # Feature importance
    feature_importance = get_feature_importance(model, feature_names)
    
    # Save everything
    save_model_and_pipeline(model, scaler, feature_names, feature_importance)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

