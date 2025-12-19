import React from 'react';
import { AlertTriangle, CheckCircle, AlertCircle, TrendingUp, Activity } from 'lucide-react';
import { GaugeChart } from '@/components/ui/gauge-chart';
import { Progress } from '@/components/ui/progress';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, CartesianGrid } from 'recharts';
import { cn } from '@/lib/utils';
import { PredictionResponse } from '@/lib/api';

interface RiskPredictionPanelProps {
  prediction: PredictionResponse | null;
  riskHistory: Array<{ timestamp: number; riskScore: number }>;
  className?: string;
}

export const RiskPredictionPanel: React.FC<RiskPredictionPanelProps> = ({
  prediction,
  riskHistory,
  className,
}) => {
  if (!prediction) {
    return (
      <div className={cn("card-medical p-6", className)}>
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <Activity className="w-12 h-12 text-muted-foreground mx-auto mb-4 animate-pulse" />
            <p className="text-muted-foreground">Waiting for prediction...</p>
          </div>
        </div>
      </div>
    );
  }

  const getSeverityConfig = () => {
    switch (prediction.severity) {
      case 'Critical':
        return {
          icon: AlertTriangle,
          color: 'text-destructive',
          bgColor: 'bg-destructive/10',
          borderColor: 'border-destructive/20',
          badgeColor: 'bg-destructive text-destructive-foreground',
        };
      case 'Warning':
        return {
          icon: AlertCircle,
          color: 'text-warning',
          bgColor: 'bg-warning/10',
          borderColor: 'border-warning/20',
          badgeColor: 'bg-warning text-warning-foreground',
        };
      default:
        return {
          icon: CheckCircle,
          color: 'text-success',
          bgColor: 'bg-success/10',
          borderColor: 'border-success/20',
          badgeColor: 'bg-success text-success-foreground',
        };
    }
  };

  const severityConfig = getSeverityConfig();
  const SeverityIcon = severityConfig.icon;

  // Prepare feature contributions for chart
  const featureData = Object.entries(prediction.featureContributions)
    .map(([name, value]) => ({
      name: name.replace(/([A-Z])/g, ' $1').trim(),
      value: Math.abs(value),
      sign: value >= 0 ? '+' : '-',
    }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 6);

  // Prepare risk history for trend
  const chartData = riskHistory.slice(-60).map((point, i) => ({
    time: i,
    risk: point.riskScore * 100,
  }));

  return (
    <div className={cn("space-y-6", className)}>
      {/* Risk Score Gauge */}
      <div className={cn("card-medical p-6", severityConfig.bgColor, severityConfig.borderColor, "border-2")}>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <SeverityIcon className={cn("w-5 h-5", severityConfig.color)} />
            <h3 className="font-display font-semibold text-lg text-foreground">
              Deterioration Risk
            </h3>
          </div>
          <span className={cn("px-3 py-1 rounded-full text-xs font-bold", severityConfig.badgeColor)}>
            {prediction.severity}
          </span>
        </div>
        
        <div className="flex flex-col items-center mb-4">
          <GaugeChart
            value={prediction.riskScore * 100}
            min={0}
            max={100}
            label="Risk Score"
            unit="%"
            size="lg"
            variant={prediction.severity === 'Critical' ? 'danger' : prediction.severity === 'Warning' ? 'warning' : 'success'}
          />
        </div>

        <div className="mt-4 p-3 bg-card rounded-lg border border-border">
          <p className="text-sm font-medium text-foreground mb-1">Recommended Action</p>
          <p className="text-sm text-muted-foreground">{prediction.recommendedAction}</p>
        </div>
      </div>

      {/* Risk Trend Chart */}
      {riskHistory.length > 0 && (
        <div className="card-medical p-6">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="w-5 h-5 text-primary" />
            <h3 className="font-display font-semibold text-lg text-foreground">
              Risk Trend (Last 60s)
            </h3>
          </div>
          <div className="h-32">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                <XAxis hide />
                <YAxis domain={[0, 100]} hide />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'hsl(var(--card))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '6px',
                  }}
                  formatter={(value: number) => [`${value.toFixed(1)}%`, 'Risk']}
                />
                <Bar
                  dataKey="risk"
                  fill="hsl(var(--primary))"
                  radius={[2, 2, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Feature Contributions */}
      {featureData.length > 0 && (
        <div className="card-medical p-6">
          <h3 className="font-display font-semibold text-lg text-foreground mb-4">
            Top Feature Drivers
          </h3>
          <div className="space-y-3">
            {featureData.map((feature) => (
              <div key={feature.name}>
                <div className="flex justify-between mb-1">
                  <span className="text-sm text-foreground capitalize">{feature.name}</span>
                  <span className={cn(
                    "text-sm font-medium",
                    feature.sign === '+' ? "text-destructive" : "text-success"
                  )}>
                    {feature.sign}{feature.value.toFixed(3)}
                  </span>
                </div>
                <Progress value={feature.value * 100} className="h-2" />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};





