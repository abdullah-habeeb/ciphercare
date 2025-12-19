/**
 * DataInputSection Component
 * Intelligent data input section that saves to database and displays predictions
 */

import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { cn } from '@/lib/utils';
import { 
  Database, 
  Save, 
  Play, 
  CheckCircle2, 
  Loader2,
  AlertCircle,
  TrendingUp,
  Activity
} from 'lucide-react';
import { uploadHospitalData, predictHospitalNew, getHospitalMetadata } from '@/lib/api';
import { mockPredict } from './mockModel';

export interface DataInputSectionProps {
  hospitalId: string;
  hospitalName: string;
}

// Field definitions for each hospital type
const hospitalFields: Record<string, Array<{
  name: string;
  label: string;
  type: 'number' | 'text' | 'select';
  placeholder?: string;
  options?: string[];
  min?: number;
  max?: number;
  required?: boolean;
}>> = {
  A: [
    { name: 'heart_rate', label: 'Heart Rate (BPM)', type: 'number', placeholder: '72', min: 30, max: 200, required: true },
    { name: 'systolic_bp', label: 'Systolic BP (mmHg)', type: 'number', placeholder: '120', min: 50, max: 250, required: true },
    { name: 'diastolic_bp', label: 'Diastolic BP (mmHg)', type: 'number', placeholder: '80', min: 30, max: 150, required: true },
    { name: 'resp_rate', label: 'Respiratory Rate', type: 'number', placeholder: '16', min: 8, max: 40, required: true },
    { name: 'spo2', label: 'SpO₂ (%)', type: 'number', placeholder: '98', min: 70, max: 100, required: true },
    { name: 'temperature', label: 'Temperature (°C)', type: 'number', placeholder: '36.6', min: 30, max: 45, required: true },
    { name: 'age', label: 'Age', type: 'number', placeholder: '65', min: 18, max: 120, required: true },
    { name: 'gender', label: 'Gender', type: 'select', options: ['0', '1'], placeholder: 'Select', required: true },
    { name: 'label', label: 'Deterioration Risk Label (0=Stable, 1=High Risk)', type: 'select', options: ['0', '1'], placeholder: 'Select label', required: false }
  ],
  B: [
    { name: 'heart_rate', label: 'Heart Rate (BPM)', type: 'number', placeholder: '72', min: 30, max: 200, required: true },
    { name: 'systolic_bp', label: 'Systolic BP (mmHg)', type: 'number', placeholder: '120', min: 50, max: 250, required: true },
    { name: 'diastolic_bp', label: 'Diastolic BP (mmHg)', type: 'number', placeholder: '80', min: 30, max: 150, required: true },
    { name: 'resp_rate', label: 'Respiratory Rate', type: 'number', placeholder: '16', min: 8, max: 40, required: true },
    { name: 'spo2', label: 'SpO₂ (%)', type: 'number', placeholder: '98', min: 70, max: 100, required: true },
    { name: 'temperature', label: 'Temperature (°C)', type: 'number', placeholder: '36.6', min: 30, max: 45, required: true },
    { name: 'age', label: 'Age', type: 'number', placeholder: '65', min: 18, max: 120, required: true },
    { name: 'gender', label: 'Gender', type: 'select', options: ['0', '1'], placeholder: 'Select', required: true },
    { name: 'label', label: 'Deterioration Risk Label (0=Stable, 1=High Risk)', type: 'select', options: ['0', '1'], placeholder: 'Select label', required: false }
  ],
  C: [
    { name: 'heart_rate', label: 'Heart Rate (BPM)', type: 'number', placeholder: '72', min: 30, max: 200, required: true },
    { name: 'systolic_bp', label: 'Systolic BP (mmHg)', type: 'number', placeholder: '120', min: 50, max: 250, required: true },
    { name: 'diastolic_bp', label: 'Diastolic BP (mmHg)', type: 'number', placeholder: '80', min: 30, max: 150, required: true },
    { name: 'resp_rate', label: 'Respiratory Rate', type: 'number', placeholder: '16', min: 8, max: 40, required: true },
    { name: 'spo2', label: 'SpO₂ (%)', type: 'number', placeholder: '98', min: 70, max: 100, required: true },
    { name: 'temperature', label: 'Temperature (°C)', type: 'number', placeholder: '36.6', min: 30, max: 45, required: true },
    { name: 'age', label: 'Age', type: 'number', placeholder: '65', min: 18, max: 120, required: true },
    { name: 'gender', label: 'Gender', type: 'select', options: ['0', '1'], placeholder: 'Select', required: true },
    { name: 'label', label: 'Deterioration Risk Label (0=Stable, 1=High Risk)', type: 'select', options: ['0', '1'], placeholder: 'Select label', required: false }
  ],
  D: [
    { name: 'heart_rate', label: 'Heart Rate (BPM)', type: 'number', placeholder: '72', min: 30, max: 200, required: true },
    { name: 'systolic_bp', label: 'Systolic BP (mmHg)', type: 'number', placeholder: '120', min: 50, max: 250, required: true },
    { name: 'diastolic_bp', label: 'Diastolic BP (mmHg)', type: 'number', placeholder: '80', min: 30, max: 150, required: true },
    { name: 'resp_rate', label: 'Respiratory Rate', type: 'number', placeholder: '16', min: 8, max: 40, required: true },
    { name: 'spo2', label: 'SpO₂ (%)', type: 'number', placeholder: '98', min: 70, max: 100, required: true },
    { name: 'temperature', label: 'Temperature (°C)', type: 'number', placeholder: '36.6', min: 30, max: 45, required: true },
    { name: 'age', label: 'Age', type: 'number', placeholder: '65', min: 18, max: 120, required: true },
    { name: 'gender', label: 'Gender', type: 'select', options: ['0', '1'], placeholder: 'Select', required: true },
    { name: 'label', label: 'Deterioration Risk Label (0=Stable, 1=High Risk)', type: 'select', options: ['0', '1'], placeholder: 'Select label', required: false }
  ],
  E: [
    { name: 'heart_rate', label: 'Heart Rate (BPM)', type: 'number', placeholder: '72', min: 30, max: 200, required: true },
    { name: 'systolic_bp', label: 'Systolic BP (mmHg)', type: 'number', placeholder: '120', min: 50, max: 250, required: true },
    { name: 'diastolic_bp', label: 'Diastolic BP (mmHg)', type: 'number', placeholder: '80', min: 30, max: 150, required: true },
    { name: 'resp_rate', label: 'Respiratory Rate', type: 'number', placeholder: '16', min: 8, max: 40, required: true },
    { name: 'spo2', label: 'SpO₂ (%)', type: 'number', placeholder: '98', min: 70, max: 100, required: true },
    { name: 'temperature', label: 'Temperature (°C)', type: 'number', placeholder: '36.6', min: 30, max: 45, required: true },
    { name: 'age', label: 'Age', type: 'number', placeholder: '65', min: 18, max: 120, required: true },
    { name: 'gender', label: 'Gender', type: 'select', options: ['0', '1'], placeholder: 'Select', required: true },
    { name: 'label', label: 'Deterioration Risk Label (0=Stable, 1=High Risk)', type: 'select', options: ['0', '1'], placeholder: 'Select label', required: false }
  ]
};

export const DataInputSection: React.FC<DataInputSectionProps> = ({ hospitalId, hospitalName }) => {
  const [formData, setFormData] = useState<Record<string, any>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [isSaving, setIsSaving] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionResult, setPredictionResult] = useState<any>(null);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [savedCount, setSavedCount] = useState(0);

  const fields = useMemo(() => {
    return hospitalFields[hospitalId] || hospitalFields['A'];
  }, [hospitalId]);

  const validateField = (field: typeof fields[0], value: any): string | null => {
    if (field.required && (!value || value === '')) {
      return `${field.label} is required`;
    }
    
    if (field.type === 'number' && value !== '' && value !== undefined) {
      const numValue = parseFloat(value);
      if (isNaN(numValue)) {
        return `${field.label} must be a number`;
      }
      if (field.min !== undefined && numValue < field.min) {
        return `${field.label} must be at least ${field.min}`;
      }
      if (field.max !== undefined && numValue > field.max) {
        return `${field.label} must be at most ${field.max}`;
      }
    }
    
    return null;
  };

  const handleFieldChange = (name: string, value: any) => {
    setFormData(prev => ({ ...prev, [name]: value }));
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => {
        const newErrors = { ...prev };
        delete newErrors[name];
        return newErrors;
      });
    }
    // Clear success/error messages
    setSaveSuccess(false);
    setSaveError(null);
  };

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};
    fields.forEach(field => {
      const error = validateField(field, formData[field.name]);
      if (error) {
        newErrors[field.name] = error;
      }
    });
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSaveToDatabase = async () => {
    if (!validateForm()) {
      return;
    }

    setIsSaving(true);
    setSaveError(null);
    setSaveSuccess(false);

    try {
      // Prepare data for database
      const dataToSave = { ...formData };
      
      // Convert string numbers to actual numbers
      fields.forEach(field => {
        if (field.type === 'number' && dataToSave[field.name] !== undefined) {
          dataToSave[field.name] = parseFloat(dataToSave[field.name]);
        }
      });

      // Convert label to number if present
      if (dataToSave.label !== undefined) {
        dataToSave.label = parseInt(dataToSave.label);
      }

      // Save to database
      const labelValue = dataToSave.label !== undefined ? parseInt(dataToSave.label) : undefined;
      const response = await uploadHospitalData(hospitalId, undefined, [dataToSave], labelValue);
      
      setSaveSuccess(true);
      setSavedCount(prev => prev + 1);
      
      // Clear form after successful save
      setTimeout(() => {
        setFormData({});
        setSaveSuccess(false);
      }, 3000);
    } catch (error: any) {
      setSaveError(error.message || 'Failed to save data to database');
    } finally {
      setIsSaving(false);
    }
  };

  const handlePredict = async () => {
    if (!validateForm()) {
      return;
    }

    setIsPredicting(true);
    setPredictionResult(null);

    try {
      // Prepare data for prediction
      const predictionData: Record<string, any> = {};
      
      // Map form data to prediction format
      fields.forEach(field => {
        if (formData[field.name] !== undefined && field.name !== 'label') {
          if (field.type === 'number') {
            predictionData[field.name] = parseFloat(formData[field.name]);
          } else {
            predictionData[field.name] = formData[field.name];
          }
        }
      });

      // Try to use backend prediction, fallback to mock
      let result;
      try {
        result = await predictHospitalNew(hospitalId, predictionData);
      } catch {
        // Fallback to mock prediction
        const mockResult = mockPredict(hospitalId, predictionData);
        result = {
          risk_score: mockResult.risk,
          severity: mockResult.severity,
          explanation: { condition: mockResult.condition, explanation: mockResult.explanation },
          recommended_action: mockResult.recommended_action
        };
      }

      setPredictionResult(result);
    } catch (error: any) {
      setPredictionResult({
        error: error.message || 'Failed to generate prediction'
      });
    } finally {
      setIsPredicting(false);
    }
  };

  const renderField = (field: typeof fields[0]) => {
    const hasError = !!errors[field.name];
    const value = formData[field.name] ?? '';

    if (field.type === 'select') {
      return (
        <div key={field.name} className="space-y-2">
          <Label htmlFor={field.name} className={cn(hasError && "text-destructive")}>
            {field.label} {field.required && <span className="text-destructive">*</span>}
          </Label>
          <Select
            value={value}
            onValueChange={(val) => handleFieldChange(field.name, val)}
            disabled={isSaving || isPredicting}
          >
            <SelectTrigger className={cn(hasError && "border-destructive")}>
              <SelectValue placeholder={field.placeholder || `Select ${field.label}`} />
            </SelectTrigger>
            <SelectContent>
              {field.options?.map((option) => (
                <SelectItem key={option} value={option}>
                  {option === '0' && field.name === 'gender' ? 'Female' : 
                   option === '1' && field.name === 'gender' ? 'Male' :
                   option === '0' && field.name === 'label' ? 'Stable (0)' :
                   option === '1' && field.name === 'label' ? 'High Risk (1)' :
                   option}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {hasError && (
            <p className="text-sm text-destructive">{errors[field.name]}</p>
          )}
        </div>
      );
    }

    return (
      <div key={field.name} className="space-y-2">
        <Label htmlFor={field.name} className={cn(hasError && "text-destructive")}>
          {field.label} {field.required && <span className="text-destructive">*</span>}
        </Label>
        <Input
          id={field.name}
          type={field.type}
          placeholder={field.placeholder}
          value={value}
          onChange={(e) => handleFieldChange(field.name, e.target.value)}
          min={field.min}
          max={field.max}
          className={cn(hasError && "border-destructive")}
          disabled={isSaving || isPredicting}
        />
        {hasError && (
          <p className="text-sm text-destructive">{errors[field.name]}</p>
        )}
      </div>
    );
  };

  const getSeverityColor = (severity: string) => {
    switch (severity?.toLowerCase()) {
      case 'critical':
        return 'bg-destructive/20 text-destructive border-destructive/30';
      case 'high':
        return 'bg-warning/20 text-warning border-warning/30';
      case 'moderate':
        return 'bg-warning/20 text-warning border-warning/30';
      case 'low':
      case 'stable':
        return 'bg-success/20 text-success border-success/30';
      default:
        return 'bg-secondary text-foreground border-border';
    }
  };

  return (
    <div className="space-y-6">
      <Card className="card-medical">
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
              <Database className="w-5 h-5 text-primary" />
            </div>
            <div className="flex-1">
              <CardTitle>Test Data Input & Prediction</CardTitle>
              <CardDescription>
                Input patient data, save to database, and get real-time predictions for {hospitalName}
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Input Form */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-semibold text-foreground">Patient Data Input</h3>
                  {savedCount > 0 && (
                    <p className="text-xs text-muted-foreground mt-1">
                      {savedCount} record{savedCount !== 1 ? 's' : ''} saved to database
                    </p>
                  )}
                </div>
                {saveSuccess && (
                  <Badge className="bg-success/20 text-success border-success/30 animate-in fade-in">
                    <CheckCircle2 className="w-3 h-3 mr-1" />
                    Saved
                  </Badge>
                )}
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {fields.map(field => renderField(field))}
              </div>

              {saveError && (
                <div className="p-3 bg-destructive/10 border border-destructive/20 rounded-lg">
                  <div className="flex items-center gap-2">
                    <AlertCircle className="w-4 h-4 text-destructive" />
                    <p className="text-sm text-destructive">{saveError}</p>
                  </div>
                </div>
              )}

              <div className="flex gap-3 pt-2">
                <Button
                  onClick={handleSaveToDatabase}
                  disabled={isSaving || isPredicting}
                  className="flex-1"
                >
                  {isSaving ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Saving...
                    </>
                  ) : (
                    <>
                      <Save className="w-4 h-4 mr-2" />
                      Save to Database
                    </>
                  )}
                </Button>
                <Button
                  onClick={handlePredict}
                  disabled={isSaving || isPredicting}
                  variant="default"
                  className="flex-1"
                >
                  {isPredicting ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Predicting...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 mr-2" />
                      Get Prediction
                    </>
                  )}
                </Button>
              </div>
            </div>

            {/* Prediction Results */}
            <div className="space-y-4">
              <h3 className="font-semibold text-foreground">Prediction Output</h3>
              
              {isPredicting ? (
                <Card>
                  <CardContent className="flex items-center justify-center py-12">
                    <div className="text-center space-y-3">
                      <Loader2 className="w-8 h-8 animate-spin text-primary mx-auto" />
                      <p className="text-sm text-muted-foreground">Processing prediction...</p>
                    </div>
                  </CardContent>
                </Card>
              ) : predictionResult ? (
                <Card className={cn("border-2", getSeverityColor(predictionResult.severity))}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="flex items-center gap-2">
                        <Activity className="w-5 h-5" />
                        Prediction Results
                      </CardTitle>
                      <Badge className={cn(getSeverityColor(predictionResult.severity))}>
                        {predictionResult.severity?.toUpperCase() || 'UNKNOWN'}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {predictionResult.error ? (
                      <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
                        <p className="text-sm text-destructive">{predictionResult.error}</p>
                      </div>
                    ) : (
                      <>
                        <div className="space-y-2">
                          <div className="flex items-center justify-between">
                            <span className="text-sm font-medium text-muted-foreground">Risk Score</span>
                            <span className="text-2xl font-bold font-mono text-primary">
                              {Math.round((predictionResult.risk_score || 0) * 100)}%
                            </span>
                          </div>
                          <Progress 
                            value={(predictionResult.risk_score || 0) * 100} 
                            className="h-3"
                          />
                        </div>

                        {predictionResult.explanation && (
                          <div className="p-4 bg-secondary rounded-lg">
                            <p className="text-xs text-muted-foreground mb-1">Explanation</p>
                            <p className="text-sm text-foreground">
                              {typeof predictionResult.explanation === 'object' 
                                ? predictionResult.explanation.explanation || predictionResult.explanation.condition
                                : predictionResult.explanation}
                            </p>
                          </div>
                        )}

                        {predictionResult.recommended_action && (
                          <div className={cn("p-4 rounded-lg border-2", getSeverityColor(predictionResult.severity))}>
                            <p className="text-xs font-medium mb-1 opacity-80">Recommended Action</p>
                            <p className="font-semibold text-foreground">{predictionResult.recommended_action}</p>
                          </div>
                        )}
                      </>
                    )}
                  </CardContent>
                </Card>
              ) : (
                <Card>
                  <CardContent className="flex items-center justify-center py-12">
                    <div className="text-center space-y-2">
                      <TrendingUp className="w-12 h-12 text-muted-foreground mx-auto opacity-50" />
                      <p className="text-sm text-muted-foreground">
                        Enter data and click "Get Prediction" to see results
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

