import React from 'react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip, CartesianGrid } from 'recharts';
import { VitalsData } from '@/lib/iomt-simulator';
import { cn } from '@/lib/utils';
import { Heart, Wind, Thermometer, Activity } from 'lucide-react';

interface VitalsTrendChartProps {
  data: VitalsData[];
  metric: 'heartRate' | 'spo2' | 'temperature' | 'respiratoryRate';
  label: string;
  unit: string;
  color?: string;
  icon?: React.ElementType;
  className?: string;
  min?: number;
  max?: number;
}

export const VitalsTrendChart: React.FC<VitalsTrendChartProps> = ({
  data,
  metric,
  label,
  unit,
  color = 'hsl(var(--primary))',
  icon: Icon = Activity,
  className,
  min,
  max,
}) => {
  const chartData = data.map((v, i) => ({
    time: i,
    value: v[metric],
    timestamp: v.timestamp,
  }));

  const currentValue = data.length > 0 ? data[data.length - 1][metric] : 0;
  
  // Auto-calculate min/max if not provided
  const values = chartData.map(d => d.value);
  const chartMin = min ?? Math.min(...values) - (Math.max(...values) - Math.min(...values)) * 0.1;
  const chartMax = max ?? Math.max(...values) + (Math.max(...values) - Math.min(...values)) * 0.1;

  return (
    <div className={cn("card-medical p-4", className)}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Icon className="w-4 h-4 text-muted-foreground" />
          <span className="text-sm font-medium text-foreground">{label}</span>
        </div>
        <div className="text-right">
          <div className="text-lg font-bold font-mono text-foreground">
            {typeof currentValue === 'number' ? currentValue.toFixed(metric === 'temperature' ? 1 : 0) : currentValue}
            <span className="text-xs text-muted-foreground ml-1">{unit}</span>
          </div>
        </div>
      </div>
      <div className="h-32">
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
              <XAxis 
                dataKey="time" 
                hide 
                domain={['dataMin', 'dataMax']}
              />
              <YAxis 
                domain={[chartMin, chartMax]}
                hide
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'hsl(var(--card))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '6px',
                  padding: '4px 8px',
                }}
                formatter={(value: number) => [
                  `${value.toFixed(metric === 'temperature' ? 1 : 0)} ${unit}`,
                  label,
                ]}
                labelFormatter={() => ''}
              />
              <Line
                type="monotone"
                dataKey="value"
                stroke={color}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, fill: color }}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-full flex items-center justify-center text-muted-foreground text-sm">
            No data available
          </div>
        )}
      </div>
    </div>
  );
};





