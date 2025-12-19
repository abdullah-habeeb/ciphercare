/**
 * UseCaseInterface Component
 * Main component that orchestrates the hospital use-case prediction interface
 */

import React, { useState, useMemo } from 'react';
import { UseCaseForm, FormField } from './UseCaseForm';
import { UseCaseResultCard, UseCaseResult } from './UseCaseResultCard';
import { mockPredict } from './mockModel';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Stethoscope } from 'lucide-react';

export interface UseCaseInterfaceProps {
  hospitalId: string;
  hospitalName: string;
}

// Form schemas for each hospital
const hospitalSchemas: Record<string, FormField[]> = {
  A: [
    { name: 'heartRate', label: 'Heart Rate (BPM)', type: 'number', placeholder: '72', min: 30, max: 200, required: true },
    { name: 'qrsDuration', label: 'QRS Duration (ms)', type: 'number', placeholder: '80', min: 40, max: 200, required: true },
    { name: 'qtInterval', label: 'QT Interval (ms)', type: 'number', placeholder: '400', min: 300, max: 600, required: true },
    { name: 'stElevation', label: 'ST Elevation (mm)', type: 'number', placeholder: '0', min: 0, max: 10, required: true },
    { name: 'age', label: 'Age', type: 'number', placeholder: '65', min: 18, max: 120, required: true }
  ],
  B: [
    { name: 'hr', label: 'Heart Rate (BPM)', type: 'number', placeholder: '72', min: 30, max: 200, required: true },
    { name: 'bpSystolic', label: 'BP Systolic (mmHg)', type: 'number', placeholder: '120', min: 50, max: 250, required: true },
    { name: 'bpDiastolic', label: 'BP Diastolic (mmHg)', type: 'number', placeholder: '80', min: 30, max: 150, required: true },
    { name: 'spo2', label: 'SpO₂ (%)', type: 'number', placeholder: '98', min: 70, max: 100, required: true },
    { name: 'respiratoryRate', label: 'Respiratory Rate (per min)', type: 'number', placeholder: '16', min: 8, max: 40, required: true },
    { name: 'temperature', label: 'Temperature (°C)', type: 'number', placeholder: '36.6', min: 30, max: 45, required: true }
  ],
  C: [
    { name: 'image', label: 'Chest X-Ray Image', type: 'file', placeholder: 'Image upload (placeholder)', required: false },
    { name: 'symptoms', label: 'Symptoms', type: 'select', placeholder: 'Select symptoms', options: ['none', 'cough', 'fever', 'shortness of breath', 'chest pain', 'fatigue'], required: true },
    { name: 'smokingHistory', label: 'Smoking History', type: 'select', placeholder: 'Select history', options: ['no', 'yes', 'former'], required: true },
    { name: 'coughDuration', label: 'Cough Duration (days)', type: 'number', placeholder: '0', min: 0, max: 30, required: true }
  ],
  D: [
    { name: 'heartRate', label: 'Heart Rate (BPM)', type: 'number', placeholder: '72', min: 30, max: 200, required: true },
    { name: 'variabilityIndex', label: 'Variability Index', type: 'number', placeholder: '30', min: 0, max: 100, required: true },
    { name: 'frailtyScore', label: 'Frailty Score', type: 'number', placeholder: '3', min: 0, max: 10, required: true },
    { name: 'medicationLoad', label: 'Medication Load', type: 'number', placeholder: '3', min: 0, max: 15, required: true }
  ],
  E: [
    { name: 'hr', label: 'Heart Rate (BPM)', type: 'number', placeholder: '72', min: 30, max: 200, required: true },
    { name: 'spo2', label: 'SpO₂ (%)', type: 'number', placeholder: '98', min: 70, max: 100, required: true },
    { name: 'coughDuration', label: 'Cough Duration (days)', type: 'number', placeholder: '0', min: 0, max: 30, required: true },
    { name: 'xrayDescription', label: 'X-Ray Description', type: 'text', placeholder: 'Brief description of findings', required: true },
    { name: 'age', label: 'Age', type: 'number', placeholder: '65', min: 18, max: 120, required: true },
    { name: 'comorbidities', label: 'Number of Comorbidities', type: 'number', placeholder: '0', min: 0, max: 10, required: true }
  ]
};

export const UseCaseInterface: React.FC<UseCaseInterfaceProps> = ({ hospitalId, hospitalName }) => {
  const [result, setResult] = useState<UseCaseResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const schema = useMemo(() => {
    return hospitalSchemas[hospitalId] || hospitalSchemas['A'];
  }, [hospitalId]);

  const handleSubmit = async (formData: Record<string, any>) => {
    setIsLoading(true);
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 800));
    
    // Convert form data to proper types
    const processedData: Record<string, any> = {};
    schema.forEach(field => {
      const value = formData[field.name];
      if (value !== undefined && value !== '') {
        if (field.type === 'number') {
          processedData[field.name] = parseFloat(value);
        } else {
          processedData[field.name] = value;
        }
      }
    });
    
    // Get mock prediction
    const prediction = mockPredict(hospitalId, processedData);
    setResult(prediction);
    setIsLoading(false);
  };

  return (
    <div className="space-y-6">
      <Card className="card-medical">
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
              <Stethoscope className="w-5 h-5 text-primary" />
            </div>
            <div>
              <CardTitle>Hospital Use-Case Interface</CardTitle>
              <CardDescription>
                Specialty-specific prediction panel for {hospitalName}
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Form Section */}
            <div>
              <h3 className="font-semibold text-foreground mb-4">Input Parameters</h3>
              <UseCaseForm 
                schema={schema} 
                onSubmit={handleSubmit}
                isLoading={isLoading}
              />
            </div>

            {/* Results Section */}
            <div>
              <h3 className="font-semibold text-foreground mb-4">Prediction Results</h3>
              {isLoading ? (
                <Card>
                  <CardContent className="flex items-center justify-center py-12">
                    <div className="text-center space-y-3">
                      <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto" />
                      <p className="text-sm text-muted-foreground">Processing prediction...</p>
                    </div>
                  </CardContent>
                </Card>
              ) : result ? (
                <UseCaseResultCard result={result} />
              ) : (
                <Card>
                  <CardContent className="flex items-center justify-center py-12">
                    <div className="text-center space-y-2">
                      <Stethoscope className="w-12 h-12 text-muted-foreground mx-auto opacity-50" />
                      <p className="text-sm text-muted-foreground">
                        Enter parameters and run prediction to see results
                      </p>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

