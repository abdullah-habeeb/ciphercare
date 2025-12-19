import React from 'react';
import { 
  BarChart3, 
  Grid3X3, 
  TrendingUp, 
  Layers, 
  GitBranch,
  Zap,
  ArrowRight
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  LineChart,
  Line,
  Legend
} from 'recharts';
import { HOSPITALS, FL_ROUNDS_DATA } from '@/lib/constants';
import { cn } from '@/lib/utils';

const classWiseAuroc = [
  { class: 'Normal Sinus', auroc: 0.97, support: 12340 },
  { class: 'Atrial Fib', auroc: 0.94, support: 4521 },
  { class: 'Bradycardia', auroc: 0.91, support: 3890 },
  { class: 'Tachycardia', auroc: 0.93, support: 5120 },
  { class: 'ST Elevation', auroc: 0.89, support: 2341 },
  { class: 'PVC', auroc: 0.92, support: 4102 },
];

const confusionMatrix = [
  [892, 23, 12, 8, 5, 7],
  [18, 467, 8, 5, 3, 2],
  [15, 6, 412, 7, 2, 4],
  [9, 4, 11, 523, 6, 3],
  [7, 2, 3, 8, 287, 4],
  [12, 3, 5, 4, 7, 421],
];

const classes = ['Normal', 'AFib', 'Brady', 'Tachy', 'STEMI', 'PVC'];

const crossDomainData = [
  { hospital: 'Hospital A', before: 0.82, after: 0.93, gain: '+13.4%' },
  { hospital: 'Hospital B', before: 0.79, after: 0.91, gain: '+15.2%' },
  { hospital: 'Hospital C', before: 0.84, after: 0.95, gain: '+13.1%' },
  { hospital: 'Hospital D', before: 0.76, after: 0.89, gain: '+17.1%' },
  { hospital: 'Hospital E', before: 0.86, after: 0.96, gain: '+11.6%' },
];

const modelSizeData = FL_ROUNDS_DATA.map((r) => ({
  round: r.round,
  size: 24.5 + r.round * 0.3 + Math.random() * 0.2,
  accuracy: r.globalAuroc,
}));

const ModelInsights: React.FC = () => {
  const maxVal = Math.max(...confusionMatrix.flat());

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div>
        <h1 className="text-2xl md:text-3xl font-display font-bold text-foreground flex items-center gap-3">
          <BarChart3 className="w-8 h-8 text-primary" />
          Global Model Insights
        </h1>
        <p className="text-muted-foreground mt-1">
          Deep analytics and performance breakdown of the federated model
        </p>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-4 gap-4">
        <div className="card-medical p-4">
          <p className="text-sm text-muted-foreground">Global AUROC</p>
          <p className="text-3xl font-bold font-mono text-primary">0.942</p>
          <p className="text-xs text-success">+20.5% vs baseline</p>
        </div>
        <div className="card-medical p-4">
          <p className="text-sm text-muted-foreground">Macro F1</p>
          <p className="text-3xl font-bold font-mono text-foreground">0.918</p>
          <p className="text-xs text-muted-foreground">6-class average</p>
        </div>
        <div className="card-medical p-4">
          <p className="text-sm text-muted-foreground">Model Size</p>
          <p className="text-3xl font-bold font-mono text-foreground">27.8</p>
          <p className="text-xs text-muted-foreground">MB parameters</p>
        </div>
        <div className="card-medical p-4">
          <p className="text-sm text-muted-foreground">Training Rounds</p>
          <p className="text-3xl font-bold font-mono text-foreground">10</p>
          <p className="text-xs text-success">Complete</p>
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Confusion Matrix */}
        <div className="card-medical p-6">
          <div className="flex items-center gap-2 mb-4">
            <Grid3X3 className="w-5 h-5 text-primary" />
            <h3 className="font-display font-semibold text-lg text-foreground">Confusion Matrix</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr>
                  <th className="p-2"></th>
                  {classes.map(c => (
                    <th key={c} className="p-2 text-center text-xs text-muted-foreground font-medium">{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {confusionMatrix.map((row, i) => (
                  <tr key={i}>
                    <td className="p-2 text-xs text-muted-foreground font-medium">{classes[i]}</td>
                    {row.map((val, j) => {
                      const intensity = val / maxVal;
                      const isDiagonal = i === j;
                      return (
                        <td 
                          key={j} 
                          className={cn(
                            "p-2 text-center font-mono text-xs rounded",
                            isDiagonal 
                              ? "bg-success/20 text-success font-bold" 
                              : val > 10 
                                ? "bg-destructive/20 text-destructive" 
                                : "bg-muted text-muted-foreground"
                          )}
                        >
                          {val}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Class-wise AUROC */}
        <div className="card-medical p-6">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="w-5 h-5 text-primary" />
            <h3 className="font-display font-semibold text-lg text-foreground">Class-wise AUROC</h3>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={classWiseAuroc} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis type="number" domain={[0.85, 1]} stroke="hsl(var(--muted-foreground))" fontSize={11} />
                <YAxis dataKey="class" type="category" stroke="hsl(var(--muted-foreground))" fontSize={11} width={80} />
                <Tooltip 
                  contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '8px' }}
                  formatter={(value: number) => [value.toFixed(3), 'AUROC']}
                />
                <Bar dataKey="auroc" fill="hsl(var(--primary))" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Cross-Domain Benefit */}
        <div className="card-medical p-6">
          <div className="flex items-center gap-2 mb-4">
            <GitBranch className="w-5 h-5 text-primary" />
            <h3 className="font-display font-semibold text-lg text-foreground">Cross-Domain Benefits</h3>
          </div>
          <p className="text-sm text-muted-foreground mb-4">
            Performance improvement after federated aggregation with diverse data sources
          </p>
          <div className="space-y-3">
            {crossDomainData.map((item) => (
              <div key={item.hospital} className="flex items-center gap-4">
                <span className="w-24 text-sm text-foreground">{item.hospital}</span>
                <div className="flex-1 flex items-center gap-2">
                  <div className="w-16 text-right">
                    <span className="text-sm font-mono text-muted-foreground">{item.before.toFixed(2)}</span>
                  </div>
                  <ArrowRight className="w-4 h-4 text-primary" />
                  <div className="w-16">
                    <span className="text-sm font-mono font-bold text-primary">{item.after.toFixed(2)}</span>
                  </div>
                  <span className="text-xs font-medium text-success bg-success/10 px-2 py-0.5 rounded">
                    {item.gain}
                  </span>
                </div>
              </div>
            ))}
          </div>
          <div className="mt-4 p-3 bg-primary/5 rounded-lg border border-primary/20">
            <p className="text-xs text-muted-foreground">
              <strong className="text-foreground">Key insight:</strong> Hospital D saw the largest improvement (+17.1%) 
              after ECG-rich Hospital A contributed to the federation.
            </p>
          </div>
        </div>

        {/* Model Size Over Rounds */}
        <div className="card-medical p-6">
          <div className="flex items-center gap-2 mb-4">
            <Layers className="w-5 h-5 text-primary" />
            <h3 className="font-display font-semibold text-lg text-foreground">Model Evolution</h3>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={modelSizeData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="round" stroke="hsl(var(--muted-foreground))" fontSize={11} tickFormatter={(v) => `R${v}`} />
                <YAxis yAxisId="left" stroke="hsl(var(--muted-foreground))" fontSize={11} />
                <YAxis yAxisId="right" orientation="right" stroke="hsl(var(--muted-foreground))" fontSize={11} domain={[0.75, 1]} />
                <Tooltip contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '8px' }} />
                <Legend />
                <Line yAxisId="left" type="monotone" dataKey="size" stroke="hsl(var(--accent))" strokeWidth={2} name="Size (MB)" />
                <Line yAxisId="right" type="monotone" dataKey="accuracy" stroke="hsl(var(--primary))" strokeWidth={2} name="AUROC" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Fine-tuning Impact */}
      <div className="card-medical p-6">
        <div className="flex items-center gap-2 mb-4">
          <Zap className="w-5 h-5 text-primary" />
          <h3 className="font-display font-semibold text-lg text-foreground">Personalization Impact by Hospital</h3>
        </div>
        <div className="grid grid-cols-5 gap-4">
          {HOSPITALS.map((hospital) => (
            <div key={hospital.id} className="p-4 bg-secondary rounded-xl text-center">
              <p className="font-semibold text-foreground mb-2">{hospital.name}</p>
              <p className="text-2xl font-bold font-mono text-primary">
                +{(hospital.personalizedBoost * 100).toFixed(1)}%
              </p>
              <p className="text-xs text-muted-foreground mt-1">from fine-tuning</p>
              <div className="mt-2 pt-2 border-t border-border">
                <p className="text-xs text-muted-foreground">Final AUROC</p>
                <p className="text-sm font-mono font-bold text-success">
                  {(hospital.globalAuroc + hospital.personalizedBoost).toFixed(3)}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ModelInsights;
