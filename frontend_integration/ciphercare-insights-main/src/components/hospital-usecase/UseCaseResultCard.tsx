/**
 * UseCaseResultCard Component
 * Displays prediction results with severity badges, risk gauge, and recommendations
 */

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { cn } from '@/lib/utils';
import { 
  AlertCircle, 
  CheckCircle2, 
  AlertTriangle, 
  XCircle,
  Activity,
  FileText,
  Lightbulb
} from 'lucide-react';

export interface UseCaseResult {
  risk: number;
  severity: 'low' | 'moderate' | 'high' | 'critical';
  condition: string;
  explanation: string;
  recommended_action: string;
}

export interface UseCaseResultCardProps {
  result: UseCaseResult;
}

const severityConfig = {
  low: {
    color: 'bg-success/20 text-success border-success/30',
    icon: CheckCircle2,
    label: 'Low Risk',
    bgColor: 'bg-success/5',
    progressColor: 'bg-success'
  },
  moderate: {
    color: 'bg-warning/20 text-warning border-warning/30',
    icon: AlertTriangle,
    label: 'Moderate Risk',
    bgColor: 'bg-warning/5',
    progressColor: 'bg-warning'
  },
  high: {
    color: 'bg-warning/20 text-warning border-warning/30',
    icon: AlertCircle,
    label: 'High Risk',
    bgColor: 'bg-warning/5',
    progressColor: 'bg-warning'
  },
  critical: {
    color: 'bg-destructive/20 text-destructive border-destructive/30',
    icon: XCircle,
    label: 'Critical Risk',
    bgColor: 'bg-destructive/5',
    progressColor: 'bg-destructive'
  }
};

export const UseCaseResultCard: React.FC<UseCaseResultCardProps> = ({ result }) => {
  const config = severityConfig[result.severity];
  const Icon = config.icon;
  const riskPercentage = Math.round(result.risk * 100);

  return (
    <div className="space-y-4 animate-in slide-in-from-bottom-4 duration-500">
      <Card className={cn("border-2", config.color)}>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Icon className="w-5 h-5" />
              Prediction Results
            </CardTitle>
            <Badge className={cn("text-xs font-semibold", config.color)}>
              {config.label}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Risk Score with Radial Gauge */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-muted-foreground">Risk Score</span>
              <span className={cn("text-2xl font-bold font-mono", config.color.replace('bg-', 'text-').replace('/20', ''))}>
                {riskPercentage}%
              </span>
            </div>
            <div className="relative">
              <Progress 
                value={riskPercentage} 
                className={cn("h-3", config.progressColor)}
              />
            </div>
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>0%</span>
              <span>50%</span>
              <span>100%</span>
            </div>
          </div>

          {/* Condition */}
          <div className={cn("p-4 rounded-lg border", config.bgColor, config.color.replace('bg-', 'border-').replace('/20', '/20'))}>
            <div className="flex items-start gap-3">
              <Activity className="w-5 h-5 mt-0.5 flex-shrink-0" />
              <div className="flex-1">
                <p className="text-xs text-muted-foreground mb-1">Detected Condition</p>
                <p className="font-semibold text-foreground">{result.condition}</p>
              </div>
            </div>
          </div>

          {/* Explanation */}
          <div className="p-4 bg-secondary rounded-lg">
            <div className="flex items-start gap-3">
              <FileText className="w-5 h-5 mt-0.5 flex-shrink-0 text-muted-foreground" />
              <div className="flex-1">
                <p className="text-xs text-muted-foreground mb-1">Explanation</p>
                <p className="text-sm text-foreground">{result.explanation}</p>
              </div>
            </div>
          </div>

          {/* Recommended Action */}
          <div className={cn("p-4 rounded-lg border-2", config.color)}>
            <div className="flex items-start gap-3">
              <Lightbulb className={cn("w-5 h-5 mt-0.5 flex-shrink-0", config.color.replace('bg-', 'text-').replace('/20', ''))} />
              <div className="flex-1">
                <p className="text-xs font-medium mb-1 opacity-80">Recommended Action</p>
                <p className="font-semibold text-foreground">{result.recommended_action}</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

