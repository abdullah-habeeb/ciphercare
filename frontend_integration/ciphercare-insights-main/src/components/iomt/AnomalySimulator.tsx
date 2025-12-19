import React from 'react';
import { AlertTriangle, Zap, Heart, Wind, Thermometer, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { AnomalyConfig } from '@/lib/iomt-simulator';
import { cn } from '@/lib/utils';

interface AnomalySimulatorProps {
  currentAnomaly: AnomalyConfig;
  onTriggerAnomaly: (type: AnomalyConfig['type'], intensity: number, duration: number) => void;
  onClearAnomaly: () => void;
  className?: string;
}

const anomalyTypes: Array<{
  type: AnomalyConfig['type'];
  label: string;
  icon: React.ElementType;
  color: string;
  description: string;
}> = [
  {
    type: 'arrhythmia',
    label: 'Arrhythmia',
    icon: Heart,
    color: 'text-destructive',
    description: 'Irregular heart rhythm',
  },
  {
    type: 'tachycardia',
    label: 'Tachycardia',
    icon: Zap,
    color: 'text-warning',
    description: 'Elevated heart rate (>120 BPM)',
  },
  {
    type: 'bradycardia',
    label: 'Bradycardia',
    icon: Heart,
    color: 'text-primary',
    description: 'Low heart rate (<60 BPM)',
  },
  {
    type: 'hypoxia',
    label: 'Hypoxia',
    icon: Wind,
    color: 'text-destructive',
    description: 'Low oxygen saturation (<90%)',
  },
  {
    type: 'fever',
    label: 'Fever',
    icon: Thermometer,
    color: 'text-warning',
    description: 'Elevated temperature (>38.5°C)',
  },
];

export const AnomalySimulator: React.FC<AnomalySimulatorProps> = ({
  currentAnomaly,
  onTriggerAnomaly,
  onClearAnomaly,
  className,
}) => {
  return (
    <div className={cn("card-medical p-6", className)}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <AlertTriangle className="w-5 h-5 text-warning" />
          <h3 className="font-display font-semibold text-lg text-foreground">Anomaly Simulator</h3>
        </div>
        {currentAnomaly.type !== 'none' && (
          <Button
            variant="outline"
            size="sm"
            onClick={onClearAnomaly}
            className="gap-2"
          >
            <X className="w-4 h-4" />
            Clear
          </Button>
        )}
      </div>

      {currentAnomaly.type !== 'none' && (
        <div className="mb-4 p-3 bg-warning/10 border border-warning/20 rounded-lg">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-warning" />
            <span className="text-sm font-medium text-foreground">
              Active: {anomalyTypes.find(a => a.type === currentAnomaly.type)?.label}
            </span>
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            Intensity: {(currentAnomaly.intensity * 100).toFixed(0)}% • 
            Duration: {currentAnomaly.duration}s
          </p>
        </div>
      )}

      <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
        {anomalyTypes.map((anomaly) => {
          const Icon = anomaly.icon;
          const isActive = currentAnomaly.type === anomaly.type;
          
          return (
            <Button
              key={anomaly.type}
              variant={isActive ? 'default' : 'outline'}
              size="sm"
              onClick={() => onTriggerAnomaly(anomaly.type, 0.7, 30)}
              className={cn(
                "flex flex-col items-center gap-1 h-auto py-3",
                isActive && "bg-warning text-warning-foreground"
              )}
            >
              <Icon className={cn("w-4 h-4", isActive ? "text-warning-foreground" : anomaly.color)} />
              <span className="text-xs font-medium">{anomaly.label}</span>
            </Button>
          );
        })}
      </div>

      <p className="text-xs text-muted-foreground mt-4 text-center">
        Click any anomaly type to simulate it in real-time
      </p>
    </div>
  );
};





