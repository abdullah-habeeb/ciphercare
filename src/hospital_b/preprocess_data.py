import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
import os

DATA_DIR = r"c:\Users\aishw\codered5\data\hospital_b"
OUTPUT_FILE = os.path.join(DATA_DIR, "processed_vitals.csv")
NUM_PATIENTS = 1000  # Simulated dataset size

def generate_synthetic_mimic_data():
    """Generate synthetic MIMIC-IV-like tabular data for Hospital B"""
    np.random.seed(42)
    
    # Patient demographics
    patient_ids = [f"P{i:05d}" for i in range(NUM_PATIENTS)]
    ages = np.random.normal(65, 15, NUM_PATIENTS).clip(18, 95)
    genders = np.random.choice(['M', 'F'], NUM_PATIENTS)
    ethnicities = np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], NUM_PATIENTS)
    
    # Vitals (with hypertension/diabetes skew)
    heart_rate = np.random.normal(85, 15, NUM_PATIENTS).clip(40, 180)
    systolic_bp = np.random.normal(145, 25, NUM_PATIENTS).clip(80, 220)  # Skewed high
    diastolic_bp = np.random.normal(88, 15, NUM_PATIENTS).clip(50, 130)
    respiratory_rate = np.random.normal(18, 4, NUM_PATIENTS).clip(8, 40)
    spo2 = np.random.normal(96, 3, NUM_PATIENTS).clip(70, 100)
    temperature = np.random.normal(37.0, 0.8, NUM_PATIENTS).clip(35, 41)
    
    # Lab values
    glucose = np.random.normal(160, 50, NUM_PATIENTS).clip(60, 400)  # Diabetes skew
    creatinine = np.random.normal(1.2, 0.6, NUM_PATIENTS).clip(0.5, 5)
    wbc = np.random.normal(9, 3, NUM_PATIENTS).clip(2, 25)
    hemoglobin = np.random.normal(12.5, 2, NUM_PATIENTS).clip(6, 18)
    
    # Comorbidities
    has_hypertension = np.random.binomial(1, 0.6, NUM_PATIENTS)
    has_diabetes = np.random.binomial(1, 0.5, NUM_PATIENTS)
    
    # Target: Clinical deterioration risk (based on vitals)
    # High risk if: low SpO2, high RR, extreme BP, high glucose
    risk_score = (
        (spo2 < 92) * 0.3 +
        (respiratory_rate > 24) * 0.25 +
        (systolic_bp > 180) * 0.2 +
        (glucose > 250) * 0.15 +
        (heart_rate > 120) * 0.1
    )
    deterioration_risk = (risk_score > 0.4).astype(int)
    
    # Introduce some missing values (realistic)
    def add_missing(arr, rate=0.05):
        mask = np.random.random(len(arr)) < rate
        arr = arr.copy()
        arr[mask] = np.nan
        return arr
    
    df = pd.DataFrame({
        'patient_id': patient_ids,
        'age': ages,
        'gender': genders,
        'ethnicity': ethnicities,
        'heart_rate': add_missing(heart_rate),
        'systolic_bp': add_missing(systolic_bp),
        'diastolic_bp': add_missing(diastolic_bp),
        'respiratory_rate': add_missing(respiratory_rate),
        'spo2': add_missing(spo2),
        'temperature': add_missing(temperature),
        'glucose': add_missing(glucose),
        'creatinine': add_missing(creatinine),
        'wbc': add_missing(wbc),
        'hemoglobin': add_missing(hemoglobin),
        'has_hypertension': has_hypertension,
        'has_diabetes': has_diabetes,
        'deterioration_risk': deterioration_risk
    })
    
    return df

def preprocess_data(df):
    """Preprocess: impute, normalize, encode"""
    
    # Separate features and target
    target = df['deterioration_risk']
    patient_ids = df['patient_id']
    
    # Encode categorical
    le_gender = LabelEncoder()
    le_ethnicity = LabelEncoder()
    
    df['gender_encoded'] = le_gender.fit_transform(df['gender'])
    df['ethnicity_encoded'] = le_ethnicity.fit_transform(df['ethnicity'])
    
    # Select numerical features
    numerical_cols = [
        'age', 'heart_rate', 'systolic_bp', 'diastolic_bp',
        'respiratory_rate', 'spo2', 'temperature', 'glucose',
        'creatinine', 'wbc', 'hemoglobin', 'has_hypertension',
        'has_diabetes', 'gender_encoded', 'ethnicity_encoded'
    ]
    
    X = df[numerical_cols].copy()
    
    # Impute missing values with KNN
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)
    
    # Normalize
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_imputed)
    
    # Create final dataframe
    feature_names = numerical_cols
    processed_df = pd.DataFrame(X_normalized, columns=feature_names)
    processed_df['patient_id'] = patient_ids.values
    processed_df['deterioration_risk'] = target.values
    
    return processed_df, scaler, imputer

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    print("Generating synthetic MIMIC-IV data for Hospital B...")
    raw_df = generate_synthetic_mimic_data()
    
    print("Preprocessing data (imputation, normalization, encoding)...")
    processed_df, scaler, imputer = preprocess_data(raw_df)
    
    # Save
    processed_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(processed_df)} processed patient records to {OUTPUT_FILE}")
    print(f"Deterioration risk distribution: {processed_df['deterioration_risk'].value_counts().to_dict()}")
