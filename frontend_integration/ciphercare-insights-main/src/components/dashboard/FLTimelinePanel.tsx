import React from 'react';
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  BarChart,
  Bar,
  Legend,
  ComposedChart,
  Line
} from 'recharts';
import { TrendingUp, Layers } from 'lucide-react';
import { FL_ROUNDS_DATA, HOSPITALS } from '@/lib/constants';

const stackedWeightData = FL_ROUNDS_DATA.map((round, i) => ({
  round: round.round,
  'Hospital A': 0.23 + Math.random() * 0.02,
  'Hospital B': 0.18 + Math.random() * 0.02,
  'Hospital C': 0.21 + Math.random() * 0.02,
  'Hospital D': 0.15 + Math.random() * 0.02,
  'Hospital E': 0.23 + Math.random() * 0.02,
}));

export const FLTimelinePanel: React.FC = () => {
  return (
    <div className="card-medical p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="font-display font-semibold text-lg text-foreground flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-primary" />
            Global FL Timeline
          </h3>
          <p className="text-sm text-muted-foreground">Model performance across training rounds</p>
        </div>
        <div className="flex items-center gap-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-primary" />
            <span className="text-muted-foreground">AUROC</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-gradient-to-r from-medical-red to-primary" />
            <span className="text-muted-foreground">Weights</span>
          </div>
        </div>
      </div>
      
      {/* Main AUROC Chart */}
      <div className="h-64 mb-6">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={FL_ROUNDS_DATA}>
            <defs>
              <linearGradient id="aurocGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.4}/>
                <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis 
              dataKey="round" 
              stroke="hsl(var(--muted-foreground))"
              tickFormatter={(v) => `R${v}`}
              fontSize={12}
            />
            <YAxis 
              domain={[0.75, 1]} 
              stroke="hsl(var(--muted-foreground))"
              fontSize={12}
              tickFormatter={(v) => v.toFixed(2)}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: 'hsl(var(--card))', 
                border: '1px solid hsl(var(--border))',
                borderRadius: '8px',
              }}
              formatter={(value: number) => [value.toFixed(3), 'AUROC']}
            />
            <Area 
              type="monotone" 
              dataKey="globalAuroc" 
              stroke="hsl(var(--primary))" 
              fill="url(#aurocGradient)"
              strokeWidth={3}
            />
            <Line 
              type="monotone" 
              dataKey="globalAuroc" 
              stroke="hsl(var(--primary))" 
              strokeWidth={3}
              dot={{ fill: 'hsl(var(--primary))', strokeWidth: 2, r: 5 }}
              activeDot={{ r: 8, fill: 'hsl(var(--primary))', stroke: 'hsl(var(--background))', strokeWidth: 3 }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Stacked Weight Contribution */}
      <div className="border-t border-border pt-6">
        <div className="flex items-center gap-2 mb-4">
          <Layers className="w-4 h-4 text-muted-foreground" />
          <span className="text-sm font-medium text-foreground">Weight Contribution Over Rounds</span>
        </div>
        <div className="h-40">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={stackedWeightData}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis dataKey="round" stroke="hsl(var(--muted-foreground))" fontSize={11} tickFormatter={(v) => `R${v}`} />
              <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} />
              <Tooltip contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '8px' }} />
              <Bar dataKey="Hospital A" stackId="a" fill="hsl(0 84% 60%)" />
              <Bar dataKey="Hospital B" stackId="a" fill="hsl(25 95% 53%)" />
              <Bar dataKey="Hospital C" stackId="a" fill="hsl(210 80% 50%)" />
              <Bar dataKey="Hospital D" stackId="a" fill="hsl(270 60% 55%)" />
              <Bar dataKey="Hospital E" stackId="a" fill="hsl(174 72% 40%)" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
