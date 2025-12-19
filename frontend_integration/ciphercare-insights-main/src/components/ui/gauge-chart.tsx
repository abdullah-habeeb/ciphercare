import React from 'react';
import { cn } from '@/lib/utils';

interface GaugeChartProps {
  value: number;
  min?: number;
  max?: number;
  label: string;
  unit?: string;
  size?: 'sm' | 'md' | 'lg';
  variant?: 'default' | 'success' | 'warning' | 'danger';
  showValue?: boolean;
  className?: string;
}

export const GaugeChart: React.FC<GaugeChartProps> = ({
  value,
  min = 0,
  max = 100,
  label,
  unit = '',
  size = 'md',
  variant = 'default',
  showValue = true,
  className,
}) => {
  const percentage = ((value - min) / (max - min)) * 100;
  const clampedPercentage = Math.min(Math.max(percentage, 0), 100);
  
  const sizeClasses = {
    sm: 'w-24 h-24',
    md: 'w-32 h-32',
    lg: 'w-40 h-40',
  };

  const strokeWidths = {
    sm: 6,
    md: 8,
    lg: 10,
  };

  const textSizes = {
    sm: 'text-lg',
    md: 'text-2xl',
    lg: 'text-3xl',
  };

  const radius = size === 'sm' ? 40 : size === 'md' ? 55 : 70;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (clampedPercentage / 100) * circumference * 0.75;

  const variantColors = {
    default: 'stroke-primary',
    success: 'stroke-success',
    warning: 'stroke-warning',
    danger: 'stroke-destructive',
  };

  const getAutoVariant = () => {
    if (variant !== 'default') return variant;
    if (percentage >= 80) return 'danger';
    if (percentage >= 60) return 'warning';
    return 'success';
  };

  const autoVariant = getAutoVariant();

  return (
    <div className={cn("flex flex-col items-center", className)}>
      <div className={cn("relative", sizeClasses[size])}>
        <svg className="w-full h-full transform -rotate-135" viewBox="0 0 160 160">
          {/* Background arc */}
          <circle
            cx="80"
            cy="80"
            r={radius}
            fill="none"
            strokeWidth={strokeWidths[size]}
            className="stroke-muted"
            strokeDasharray={circumference * 0.75}
            strokeDashoffset={0}
            strokeLinecap="round"
          />
          {/* Value arc */}
          <circle
            cx="80"
            cy="80"
            r={radius}
            fill="none"
            strokeWidth={strokeWidths[size]}
            className={cn("gauge-ring transition-all duration-1000", variantColors[autoVariant])}
            strokeDasharray={circumference * 0.75}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
          />
        </svg>
        {showValue && (
          <div className="absolute inset-0 flex flex-col items-center justify-center transform rotate-0">
            <span className={cn("font-bold font-mono text-foreground", textSizes[size])}>
              {value.toFixed(0)}
            </span>
            {unit && (
              <span className="text-xs text-muted-foreground uppercase">{unit}</span>
            )}
          </div>
        )}
      </div>
      <p className="mt-2 text-sm font-medium text-muted-foreground">{label}</p>
    </div>
  );
};
