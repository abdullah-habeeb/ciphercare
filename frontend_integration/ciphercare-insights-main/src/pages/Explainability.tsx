import React from 'react';
import { BrainCircuit, BarChart3, Layers, Eye } from 'lucide-react';
import { Progress } from '@/components/ui/progress';

const Explainability: React.FC = () => {
  const shapFeatures = [
    { name: 'QRS Duration', importance: 0.89, direction: 'positive' },
    { name: 'Heart Rate Variability', importance: 0.76, direction: 'positive' },
    { name: 'ST Elevation', importance: 0.68, direction: 'negative' },
    { name: 'P-Wave Amplitude', importance: 0.54, direction: 'positive' },
    { name: 'Age Factor', importance: 0.42, direction: 'negative' },
  ];

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl md:text-3xl font-display font-bold text-foreground flex items-center gap-3">
          <BrainCircuit className="w-8 h-8 text-primary" />
          Model Explainability
        </h1>
        <p className="text-muted-foreground">Understand how predictions are made</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card-medical p-6">
          <h3 className="font-display font-semibold text-lg text-foreground mb-4 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-primary" />
            SHAP Feature Importance
          </h3>
          <div className="space-y-4">
            {shapFeatures.map((feature) => (
              <div key={feature.name}>
                <div className="flex justify-between mb-1">
                  <span className="text-sm text-foreground">{feature.name}</span>
                  <span className={`text-sm font-medium ${feature.direction === 'positive' ? 'text-success' : 'text-destructive'}`}>
                    {feature.direction === 'positive' ? '+' : '-'}{(feature.importance * 100).toFixed(0)}%
                  </span>
                </div>
                <Progress value={feature.importance * 100} className="h-2" />
              </div>
            ))}
          </div>
        </div>

        <div className="card-medical p-6">
          <h3 className="font-display font-semibold text-lg text-foreground mb-4 flex items-center gap-2">
            <Eye className="w-5 h-5 text-primary" />
            Grad-CAM Visualization
          </h3>
          <div className="aspect-square bg-gradient-to-br from-medical-blue/20 via-medical-red/30 to-medical-orange/20 rounded-lg flex items-center justify-center">
            <div className="text-center">
              <Layers className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground">Heatmap overlay for X-ray analysis</p>
              <p className="text-xs text-muted-foreground mt-1">Upload an image to see activation</p>
            </div>
          </div>
        </div>

        <div className="card-medical p-6 lg:col-span-2">
          <h3 className="font-display font-semibold text-lg text-foreground mb-4">Clinical Explanation</h3>
          <div className="p-4 bg-secondary rounded-lg">
            <p className="text-foreground">
              <strong>Summary:</strong> The model identified elevated QRS duration (142ms, normal: 80-120ms) 
              and irregular heart rate variability as primary indicators. The combination of these factors 
              with ST segment changes suggests potential arrhythmia risk.
            </p>
            <p className="text-sm text-muted-foreground mt-3">
              This explanation is generated for clinical decision support only. Always consult with a qualified physician.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Explainability;
