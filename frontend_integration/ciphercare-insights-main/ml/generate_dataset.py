"""
Synthetic MIMIC-IV-style Dataset Generator for Deterioration Risk Prediction

Generates realistic patient vitals and lab data with clinical distributions.
Exports to data/patient_vitals.csv for model training.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Create data directory if it doesn't exist
# Get project root (parent of ml directory)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

def generate_synthetic_dataset(n_patients=1000):
    """
    Generate synthetic patient vitals dataset with realistic clinical distributions.
    
    Args:
        n_patients: Number of patients to generate
        
    Returns:
        DataFrame with patient vitals and deterioration risk labels
    """
    print(f"Generating synthetic dataset for {n_patients} patients...")
    
    data = []
    
    for i in range(n_patients):
        # Determine if patient is high-risk (2.2% prevalence)
        is_high_risk = np.random.random() < 0.022
        
        # Demographics
        age = np.random.normal(65, 15)
        age = max(18, min(100, age))  # Clamp to realistic range
        gender = np.random.choice([0, 1])  # 0 = Female, 1 = Male
        ethnicity = np.random.choice([0, 1, 2])  # 0 = White, 1 = Black, 2 = Other
        
        # Comorbidities (higher prevalence in high-risk patients)
        hypertension_prob = 0.4 if not is_high_risk else 0.75
        diabetes_prob = 0.25 if not is_high_risk else 0.60
        hypertension = 1 if np.random.random() < hypertension_prob else 0
        diabetes = 1 if np.random.random() < diabetes_prob else 0
        
        # Vitals - Stable patients
        if not is_high_risk:
            heart_rate = np.random.normal(72, 10)
            heart_rate = max(50, min(100, heart_rate))
            
            systolic_bp = np.random.normal(120, 15)
            systolic_bp = max(90, min(160, systolic_bp))
            
            diastolic_bp = np.random.normal(80, 10)
            diastolic_bp = max(60, min(100, diastolic_bp))
            
            resp_rate = np.random.normal(16, 3)
            resp_rate = max(12, min(22, resp_rate))
            
            spo2 = np.random.normal(98, 1.5)
            spo2 = max(94, min(100, spo2))
            
            temperature = np.random.normal(36.6, 0.4)
            temperature = max(35.5, min(37.5, temperature))
            
            # Labs - Normal ranges
            glucose = np.random.normal(100, 15)
            glucose = max(70, min(140, glucose))
            
            creatinine = np.random.normal(1.0, 0.3)
            creatinine = max(0.5, min(1.5, creatinine))
            
            wbc = np.random.normal(7.0, 2.0)
            wbc = max(4.0, min(11.0, wbc))
            
            hemoglobin = np.random.normal(14.0, 2.0)
            hemoglobin = max(10.0, min(18.0, hemoglobin))
        
        # Vitals - High-risk patients (deteriorating)
        else:
            # Tachycardia or bradycardia
            if np.random.random() < 0.6:
                heart_rate = np.random.normal(110, 15)  # Tachycardia
                heart_rate = max(100, min(150, heart_rate))
            else:
                heart_rate = np.random.normal(50, 8)  # Bradycardia
                heart_rate = max(35, min(60, heart_rate))
            
            # Hypotension or hypertension
            if np.random.random() < 0.5:
                systolic_bp = np.random.normal(95, 10)  # Hypotension
                systolic_bp = max(70, min(110, systolic_bp))
            else:
                systolic_bp = np.random.normal(160, 15)  # Hypertension
                systolic_bp = max(140, min(200, systolic_bp))
            
            diastolic_bp = systolic_bp * 0.65 + np.random.normal(0, 5)
            diastolic_bp = max(50, min(120, diastolic_bp))
            
            # Tachypnea or bradypnea
            if np.random.random() < 0.7:
                resp_rate = np.random.normal(24, 4)  # Tachypnea
                resp_rate = max(20, min(35, resp_rate))
            else:
                resp_rate = np.random.normal(10, 2)  # Bradypnea
                resp_rate = max(8, min(12, resp_rate))
            
            # Hypoxia
            spo2 = np.random.normal(88, 4)
            spo2 = max(82, min(94, spo2))
            
            # Fever or hypothermia
            if np.random.random() < 0.6:
                temperature = np.random.normal(38.5, 0.6)  # Fever
                temperature = max(37.8, min(40.0, temperature))
            else:
                temperature = np.random.normal(35.2, 0.5)  # Hypothermia
                temperature = max(34.0, min(35.8, temperature))
            
            # Labs - Abnormal ranges
            glucose = np.random.normal(180, 40) if diabetes else np.random.normal(140, 30)
            glucose = max(100, min(300, glucose))
            
            creatinine = np.random.normal(2.0, 0.8)  # Elevated
            creatinine = max(1.2, min(4.0, creatinine))
            
            wbc = np.random.normal(15.0, 5.0)  # Elevated (infection)
            wbc = max(12.0, min(25.0, wbc))
            
            hemoglobin = np.random.normal(10.0, 2.5)  # Low (anemia)
            hemoglobin = max(7.0, min(13.0, hemoglobin))
        
        # Add some realistic noise
        heart_rate += np.random.normal(0, 2)
        systolic_bp += np.random.normal(0, 3)
        diastolic_bp += np.random.normal(0, 2)
        resp_rate += np.random.normal(0, 1)
        spo2 += np.random.normal(0, 0.5)
        temperature += np.random.normal(0, 0.1)
        
        # Round to appropriate precision
        patient_data = {
            'patient_id': f'P{i+1:04d}',
            'age': round(age, 1),
            'gender': gender,
            'ethnicity': ethnicity,
            'heart_rate': round(heart_rate, 0),
            'systolic_bp': round(systolic_bp, 0),
            'diastolic_bp': round(diastolic_bp, 0),
            'resp_rate': round(resp_rate, 0),
            'spo2': round(spo2, 1),
            'temperature': round(temperature, 1),
            'glucose': round(glucose, 1),
            'creatinine': round(creatinine, 2),
            'wbc': round(wbc, 1),
            'hemoglobin': round(hemoglobin, 1),
            'hypertension': hypertension,
            'diabetes': diabetes,
            'deterioration_risk': 1 if is_high_risk else 0
        }
        
        data.append(patient_data)
    
    df = pd.DataFrame(data)
    
    # Print summary statistics
    print(f"\nDataset Summary:")
    print(f"Total patients: {len(df)}")
    print(f"Stable (0): {len(df[df['deterioration_risk'] == 0])}")
    print(f"High-risk (1): {len(df[df['deterioration_risk'] == 1])}")
    print(f"Imbalance ratio: {len(df[df['deterioration_risk'] == 0]) / len(df[df['deterioration_risk'] == 1]):.1f}:1")
    
    print(f"\nFeature Statistics:")
    print(df[['heart_rate', 'spo2', 'temperature', 'systolic_bp', 'resp_rate']].describe())
    
    return df

if __name__ == "__main__":
    # Generate dataset
    df = generate_synthetic_dataset(n_patients=1000)
    
    # Save to CSV
    output_path = DATA_DIR / "patient_vitals.csv"
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to {output_path}")
    print(f"Shape: {df.shape}")

