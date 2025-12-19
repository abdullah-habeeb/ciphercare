/**
 * Mock Model for Hospital Use-Case Predictions
 * Returns mock predictions based on hospital ID and input data
 */

export interface MockPredictionInput {
  [key: string]: any;
}

export interface MockPredictionOutput {
  risk: number;
  severity: 'low' | 'moderate' | 'high' | 'critical';
  condition: string;
  explanation: string;
  recommended_action: string;
}

/**
 * Mock prediction function for Hospital A - Cardiology (ECG)
 */
function predictHospitalA(inputs: MockPredictionInput): MockPredictionOutput {
  const { heartRate, qrsDuration, qtInterval, stElevation, age } = inputs;
  
  // Simple risk calculation based on inputs
  let risk = 0.5;
  
  if (heartRate && (heartRate < 60 || heartRate > 100)) risk += 0.1;
  if (qrsDuration && qrsDuration > 120) risk += 0.15;
  if (qtInterval && qtInterval > 450) risk += 0.2;
  if (stElevation && stElevation > 1) risk += 0.25;
  if (age && age > 70) risk += 0.1;
  
  risk = Math.min(1.0, risk);
  
  let severity: 'low' | 'moderate' | 'high' | 'critical' = 'low';
  if (risk >= 0.8) severity = 'critical';
  else if (risk >= 0.7) severity = 'high';
  else if (risk >= 0.5) severity = 'moderate';
  
  return {
    risk: Math.round(risk * 100) / 100,
    severity,
    condition: risk > 0.7 ? 'Possible STEMI' : risk > 0.5 ? 'Arrhythmia detected' : 'Normal rhythm',
    explanation: risk > 0.7 
      ? 'Abnormal QT and ST elevation pattern detected'
      : risk > 0.5
      ? 'Minor ECG abnormalities observed'
      : 'ECG parameters within normal range',
    recommended_action: risk > 0.7
      ? 'Immediate cardiology escalation'
      : risk > 0.5
      ? 'Schedule cardiology consultation'
      : 'Continue routine monitoring'
  };
}

/**
 * Mock prediction function for Hospital B - Vital Signs ICU
 */
function predictHospitalB(inputs: MockPredictionInput): MockPredictionOutput {
  const { hr, bpSystolic, bpDiastolic, spo2, respiratoryRate, temperature } = inputs;
  
  let risk = 0.4;
  
  if (hr && (hr < 60 || hr > 100)) risk += 0.1;
  if (bpSystolic && bpSystolic < 90) risk += 0.2;
  if (bpDiastolic && bpDiastolic < 60) risk += 0.15;
  if (spo2 && spo2 < 95) risk += 0.2;
  if (respiratoryRate && (respiratoryRate < 12 || respiratoryRate > 20)) risk += 0.15;
  if (temperature && temperature > 38) risk += 0.2;
  
  risk = Math.min(1.0, risk);
  
  let severity: 'low' | 'moderate' | 'high' | 'critical' = 'low';
  if (risk >= 0.8) severity = 'critical';
  else if (risk >= 0.6) severity = 'high';
  else if (risk >= 0.4) severity = 'moderate';
  
  return {
    risk: Math.round(risk * 100) / 100,
    severity,
    condition: risk > 0.6 ? 'Early sepsis risk' : risk > 0.4 ? 'Vital sign instability' : 'Stable vitals',
    explanation: risk > 0.6
      ? 'Low BP + high temp pattern'
      : risk > 0.4
      ? 'Some vital signs outside normal range'
      : 'All vitals within acceptable parameters',
    recommended_action: risk > 0.6
      ? 'Run blood culture + increase monitoring'
      : risk > 0.4
      ? 'Increase monitoring frequency'
      : 'Continue standard monitoring'
  };
}

/**
 * Mock prediction function for Hospital C - Radiology (Chest X-ray)
 */
function predictHospitalC(inputs: MockPredictionInput): MockPredictionOutput {
  const { symptoms, smokingHistory, coughDuration } = inputs;
  
  let risk = 0.3;
  
  if (symptoms && symptoms !== 'none') risk += 0.2;
  if (smokingHistory && smokingHistory === 'yes') risk += 0.15;
  if (coughDuration && coughDuration > 7) risk += 0.2;
  
  risk = Math.min(1.0, risk);
  
  let severity: 'low' | 'moderate' | 'high' | 'critical' = 'low';
  if (risk >= 0.7) severity = 'high';
  else if (risk >= 0.5) severity = 'moderate';
  
  return {
    risk: Math.round(risk * 100) / 100,
    severity,
    condition: risk > 0.5 ? 'Mild pulmonary congestion' : 'Normal chest imaging',
    explanation: risk > 0.5
      ? 'Symptom pattern matches low-severity cluster'
      : 'No significant abnormalities detected',
    recommended_action: risk > 0.5
      ? 'Outpatient follow-up'
      : 'Routine screening'
  };
}

/**
 * Mock prediction function for Hospital D - Geriatric ECG
 */
function predictHospitalD(inputs: MockPredictionInput): MockPredictionOutput {
  const { heartRate, variabilityIndex, frailtyScore, medicationLoad } = inputs;
  
  let risk = 0.5;
  
  if (heartRate && (heartRate < 50 || heartRate > 100)) risk += 0.15;
  if (variabilityIndex && variabilityIndex < 20) risk += 0.2;
  if (frailtyScore && frailtyScore > 5) risk += 0.2;
  if (medicationLoad && medicationLoad > 5) risk += 0.15;
  
  risk = Math.min(1.0, risk);
  
  let severity: 'low' | 'moderate' | 'high' | 'critical' = 'low';
  if (risk >= 0.8) severity = 'critical';
  else if (risk >= 0.7) severity = 'high';
  else if (risk >= 0.5) severity = 'moderate';
  
  return {
    risk: Math.round(risk * 100) / 100,
    severity,
    condition: risk > 0.7 ? 'Arrhythmia instability' : risk > 0.5 ? 'Cardiac monitoring needed' : 'Stable cardiac function',
    explanation: risk > 0.7
      ? 'High frailty and medication load with low variability'
      : risk > 0.5
      ? 'Age-related cardiac changes observed'
      : 'Cardiac parameters acceptable for age',
    recommended_action: risk > 0.7
      ? 'Admit for 24h monitoring'
      : risk > 0.5
      ? 'Increase monitoring frequency'
      : 'Continue routine care'
  };
}

/**
 * Mock prediction function for Hospital E - Multi-Modal
 */
function predictHospitalE(inputs: MockPredictionInput): MockPredictionOutput {
  const { hr, spo2, coughDuration, xrayDescription, age, comorbidities } = inputs;
  
  let risk = 0.4;
  
  if (hr && (hr < 60 || hr > 100)) risk += 0.15;
  if (spo2 && spo2 < 95) risk += 0.2;
  if (coughDuration && coughDuration > 7) risk += 0.15;
  if (xrayDescription && xrayDescription.toLowerCase().includes('opacity')) risk += 0.2;
  if (age && age > 65) risk += 0.1;
  if (comorbidities && comorbidities > 2) risk += 0.2;
  
  risk = Math.min(1.0, risk);
  
  let severity: 'low' | 'moderate' | 'high' | 'critical' = 'low';
  if (risk >= 0.8) severity = 'critical';
  else if (risk >= 0.6) severity = 'high';
  else if (risk >= 0.4) severity = 'moderate';
  
  return {
    risk: Math.round(risk * 100) / 100,
    severity,
    condition: risk > 0.8 ? 'Multi-system deterioration risk' : risk > 0.6 ? 'Multi-modal risk factors' : 'Stable multi-modal status',
    explanation: risk > 0.8
      ? 'Multiple risk factors across systems detected'
      : risk > 0.6
      ? 'Several concerning indicators present'
      : 'Multi-modal assessment within normal parameters',
    recommended_action: risk > 0.8
      ? 'Immediate ICU escalation'
      : risk > 0.6
      ? 'Close monitoring and specialist consultation'
      : 'Continue routine monitoring'
  };
}

/**
 * Main mock prediction function
 */
export function mockPredict(hospitalId: string, inputs: MockPredictionInput): MockPredictionOutput {
  switch (hospitalId) {
    case 'A':
      return predictHospitalA(inputs);
    case 'B':
      return predictHospitalB(inputs);
    case 'C':
      return predictHospitalC(inputs);
    case 'D':
      return predictHospitalD(inputs);
    case 'E':
      return predictHospitalE(inputs);
    default:
      return {
        risk: 0.5,
        severity: 'moderate',
        condition: 'Unknown condition',
        explanation: 'Hospital type not recognized',
        recommended_action: 'Consult with specialist'
      };
  }
}

