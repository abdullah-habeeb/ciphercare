import React from 'react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  Area,
  ComposedChart
} from 'recharts';
import { FL_ROUNDS_DATA } from '@/lib/constants';

export const AurocComparisonChart: React.FC = () => {
  const formattedData = FL_ROUNDS_DATA.map(d => ({
    ...d,
    globalAuroc: parseFloat(d.globalAuroc.toFixed(3)),
  }));

  return (
    <div className="card-medical p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="font-display font-semibold text-lg text-foreground">Global AUROC Progress</h3>
          <p className="text-sm text-muted-foreground">Model improvement across FL rounds</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-primary" />
            <span className="text-sm text-muted-foreground">Global</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-medical-cyan" />
            <span className="text-sm text-muted-foreground">Target</span>
          </div>
        </div>
      </div>
      
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={formattedData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="colorAuroc" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis 
              dataKey="round" 
              stroke="hsl(var(--muted-foreground))"
              fontSize={12}
              tickFormatter={(value) => `R${value}`}
            />
            <YAxis 
              domain={[0.75, 1]} 
              stroke="hsl(var(--muted-foreground))"
              fontSize={12}
              tickFormatter={(value) => value.toFixed(2)}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: 'hsl(var(--card))', 
                border: '1px solid hsl(var(--border))',
                borderRadius: '8px',
                boxShadow: 'var(--shadow-lg)'
              }}
              formatter={(value: number) => [value.toFixed(3), 'AUROC']}
              labelFormatter={(label) => `Round ${label}`}
            />
            <Area 
              type="monotone" 
              dataKey="globalAuroc" 
              stroke="hsl(var(--primary))" 
              fillOpacity={1}
              fill="url(#colorAuroc)"
              strokeWidth={3}
            />
            <Line 
              type="monotone" 
              dataKey="globalAuroc" 
              stroke="hsl(var(--primary))" 
              strokeWidth={3}
              dot={{ fill: 'hsl(var(--primary))', strokeWidth: 2, r: 4 }}
              activeDot={{ r: 6, fill: 'hsl(var(--primary))', stroke: 'hsl(var(--background))', strokeWidth: 2 }}
            />
            {/* Target line */}
            <Line 
              type="monotone" 
              dataKey={() => 0.95}
              stroke="hsl(var(--accent))" 
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={false}
              name="Target"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Before/After Comparison */}
      <div className="mt-6 grid grid-cols-2 gap-4">
        <div className="p-4 bg-secondary rounded-lg">
          <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Before FL</p>
          <p className="text-3xl font-bold font-mono text-foreground">0.782</p>
          <p className="text-sm text-muted-foreground">Baseline AUROC</p>
        </div>
        <div className="p-4 bg-gradient-to-r from-primary/10 to-primary/5 rounded-lg border border-primary/20">
          <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">After FL</p>
          <p className="text-3xl font-bold font-mono text-primary">0.942</p>
          <p className="text-sm text-success flex items-center gap-1">
            <span>+20.5% improvement</span>
          </p>
        </div>
      </div>
    </div>
  );
};
