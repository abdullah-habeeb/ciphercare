import React, { useState } from 'react';
import { 
  Network, 
  Play, 
  Pause, 
  SkipForward,
  RefreshCw,
  Info,
  ChevronRight,
  Building2,
  Server,
  ArrowRight,
  Shuffle
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { HOSPITALS, FL_ROUNDS_DATA } from '@/lib/constants';
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
  ComposedChart,
  BarChart,
  Bar
} from 'recharts';
import { cn } from '@/lib/utils';

const FederatedLearning: React.FC = () => {
  const [currentRound, setCurrentRound] = useState(8);
  const [isPlaying, setIsPlaying] = useState(false);
  const [activeTab, setActiveTab] = useState<'pipeline' | 'metrics' | 'explain'>('pipeline');

  const weightData = HOSPITALS.map(h => ({
    name: h.name,
    weight: h.contributionWeight,
    auroc: h.localAuroc,
    samples: h.samples / 1000,
  }));

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-2xl md:text-3xl font-display font-bold text-foreground flex items-center gap-3">
            <Network className="w-8 h-8 text-primary" />
            Federated Learning Pipeline
          </h1>
          <p className="text-muted-foreground mt-1">
            Visualize and control the distributed training process
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="px-4 py-2 bg-primary/10 rounded-lg border border-primary/20">
            <span className="text-sm font-medium text-primary">
              Round {currentRound} of 10
            </span>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="card-medical p-4">
        <div className="flex flex-col md:flex-row md:items-center gap-4">
          <div className="flex items-center gap-2">
            <Button 
              variant="outline" 
              size="icon"
              onClick={() => setIsPlaying(!isPlaying)}
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            </Button>
            <Button 
              variant="outline" 
              size="icon"
              onClick={() => setCurrentRound(Math.min(10, currentRound + 1))}
            >
              <SkipForward className="w-4 h-4" />
            </Button>
            <Button 
              variant="outline" 
              size="icon"
              onClick={() => setCurrentRound(1)}
            >
              <RefreshCw className="w-4 h-4" />
            </Button>
          </div>
          <div className="flex-1">
            <Slider
              value={[currentRound]}
              onValueChange={([value]) => setCurrentRound(value)}
              max={10}
              min={1}
              step={1}
              className="w-full"
            />
          </div>
          <div className="flex gap-2">
            {['pipeline', 'metrics', 'explain'].map((tab) => (
              <Button
                key={tab}
                variant={activeTab === tab ? 'default' : 'outline'}
                size="sm"
                onClick={() => setActiveTab(tab as typeof activeTab)}
                className="capitalize"
              >
                {tab}
              </Button>
            ))}
          </div>
        </div>
      </div>

      {/* Pipeline Visualization */}
      {activeTab === 'pipeline' && (
        <div className="card-medical p-6">
          <h3 className="font-display font-semibold text-lg text-foreground mb-6">
            Federated Aggregation Flow
          </h3>
          
          <div className="relative">
            {/* Hospitals Row */}
            <div className="grid grid-cols-5 gap-4 mb-8">
              {HOSPITALS.map((hospital, i) => (
                <div 
                  key={hospital.id}
                  className={cn(
                    "p-4 rounded-xl border-2 transition-all duration-500",
                    "bg-gradient-to-br from-primary/5 to-primary/10 border-primary/20",
                    isPlaying && "animate-pulse-soft"
                  )}
                  style={{ animationDelay: `${i * 100}ms` }}
                >
                  <div className="flex items-center justify-center mb-2">
                    <Building2 className="w-8 h-8 text-primary" />
                  </div>
                  <p className="text-center font-medium text-foreground text-sm">{hospital.name}</p>
                  <p className="text-center text-xs text-muted-foreground">{hospital.specialty}</p>
                  <div className="mt-2 text-center">
                    <span className="text-xs font-mono text-primary">w={hospital.contributionWeight.toFixed(2)}</span>
                  </div>
                </div>
              ))}
            </div>

            {/* Arrows */}
            <div className="flex justify-center mb-8">
              <div className="flex items-center gap-2">
                {[1, 2, 3, 4, 5].map((_, i) => (
                  <div 
                    key={i} 
                    className={cn(
                      "w-px h-8 bg-gradient-to-b from-primary/50 to-primary/20",
                      isPlaying && "animate-pulse"
                    )}
                    style={{ animationDelay: `${i * 50}ms` }}
                  />
                ))}
              </div>
            </div>

            {/* Central Server */}
            <div className="flex justify-center mb-8">
              <div className={cn(
                "p-6 rounded-2xl bg-gradient-to-br from-primary to-accent shadow-glow",
                isPlaying && "animate-pulse-soft"
              )}>
                <Server className="w-12 h-12 text-primary-foreground mx-auto mb-2" />
                <p className="text-center font-bold text-primary-foreground">Central Aggregator</p>
                <p className="text-center text-sm text-primary-foreground/80">FedProx + Weighted Avg</p>
              </div>
            </div>

            {/* Aggregation Formula */}
            <div className="flex justify-center mb-8">
              <div className="p-4 bg-secondary rounded-lg font-mono text-sm text-foreground">
                θ<sub>global</sub> = Σ w<sub>i</sub> · θ<sub>i</sub> where w<sub>i</sub> = 0.6·AUROC<sub>i</sub>² + 0.4·(n<sub>i</sub>/Σn)
              </div>
            </div>

            {/* Return Arrow */}
            <div className="flex justify-center mb-8">
              <div className="flex items-center gap-2">
                <Shuffle className="w-6 h-6 text-primary animate-spin-slow" />
                <span className="text-sm text-muted-foreground">Distribute Updated Model</span>
              </div>
            </div>

            {/* Result */}
            <div className="grid grid-cols-5 gap-4">
              {HOSPITALS.map((hospital) => (
                <div 
                  key={hospital.id}
                  className="p-3 rounded-lg bg-success/10 border border-success/20 text-center"
                >
                  <p className="text-xs text-muted-foreground">Updated</p>
                  <p className="font-mono text-sm text-success">
                    {(FL_ROUNDS_DATA[currentRound - 1]?.[`hospital${hospital.id}` as keyof typeof FL_ROUNDS_DATA[0]] || hospital.localAuroc).toFixed(3)}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Metrics Tab */}
      {activeTab === 'metrics' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Global AUROC Chart */}
          <div className="card-medical p-6">
            <h3 className="font-display font-semibold text-lg text-foreground mb-4">Global AUROC Progress</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={FL_ROUNDS_DATA.slice(0, currentRound)}>
                  <defs>
                    <linearGradient id="colorAurocFL" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="round" stroke="hsl(var(--muted-foreground))" tickFormatter={(v) => `R${v}`} />
                  <YAxis domain={[0.75, 1]} stroke="hsl(var(--muted-foreground))" />
                  <Tooltip />
                  <Area type="monotone" dataKey="globalAuroc" fill="url(#colorAurocFL)" stroke="hsl(var(--primary))" strokeWidth={2} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Per-Hospital AUROC */}
          <div className="card-medical p-6">
            <h3 className="font-display font-semibold text-lg text-foreground mb-4">Per-Hospital AUROC</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={FL_ROUNDS_DATA.slice(0, currentRound)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="round" stroke="hsl(var(--muted-foreground))" tickFormatter={(v) => `R${v}`} />
                  <YAxis domain={[0.75, 1]} stroke="hsl(var(--muted-foreground))" />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="hospitalA" stroke="hsl(0 84% 60%)" strokeWidth={2} name="Hospital A" />
                  <Line type="monotone" dataKey="hospitalB" stroke="hsl(25 95% 53%)" strokeWidth={2} name="Hospital B" />
                  <Line type="monotone" dataKey="hospitalC" stroke="hsl(210 80% 50%)" strokeWidth={2} name="Hospital C" />
                  <Line type="monotone" dataKey="hospitalD" stroke="hsl(270 60% 55%)" strokeWidth={2} name="Hospital D" />
                  <Line type="monotone" dataKey="hospitalE" stroke="hsl(174 72% 40%)" strokeWidth={2} name="Hospital E" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Contribution Weights */}
          <div className="card-medical p-6">
            <h3 className="font-display font-semibold text-lg text-foreground mb-4">Contribution Weights</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={weightData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="name" stroke="hsl(var(--muted-foreground))" />
                  <YAxis stroke="hsl(var(--muted-foreground))" />
                  <Tooltip />
                  <Bar dataKey="weight" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Drift Visualization */}
          <div className="card-medical p-6">
            <h3 className="font-display font-semibold text-lg text-foreground mb-4">FedProx Drift Analysis</h3>
            <div className="space-y-4">
              {HOSPITALS.map((hospital) => (
                <div key={hospital.id} className="flex items-center gap-4">
                  <span className="w-24 text-sm text-foreground">{hospital.name}</span>
                  <div className="flex-1 h-3 bg-secondary rounded-full overflow-hidden">
                    <div 
                      className={cn(
                        "h-full rounded-full transition-all",
                        hospital.driftMagnitude > 0.03 ? "bg-warning" : "bg-success"
                      )}
                      style={{ width: `${hospital.driftMagnitude * 1000}%` }}
                    />
                  </div>
                  <span className={cn(
                    "text-sm font-mono",
                    hospital.driftMagnitude > 0.03 ? "text-warning" : "text-success"
                  )}>
                    {hospital.driftMagnitude.toFixed(3)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Explanation Tab */}
      {activeTab === 'explain' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="card-medical p-6">
            <h3 className="font-display font-semibold text-lg text-foreground mb-4 flex items-center gap-2">
              <Info className="w-5 h-5 text-primary" />
              What is Federated Learning?
            </h3>
            <div className="prose prose-sm max-w-none">
              <p className="text-muted-foreground">
                Federated Learning (FL) is a machine learning approach that trains models across decentralized 
                data sources without exchanging raw data. Each hospital trains locally, and only model updates 
                are shared with the central server.
              </p>
              <ul className="mt-4 space-y-2 text-muted-foreground">
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-4 h-4 text-primary mt-1 flex-shrink-0" />
                  <span>Patient data never leaves the hospital</span>
                </li>
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-4 h-4 text-primary mt-1 flex-shrink-0" />
                  <span>HIPAA/GDPR compliant by design</span>
                </li>
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-4 h-4 text-primary mt-1 flex-shrink-0" />
                  <span>Combines knowledge from diverse patient populations</span>
                </li>
              </ul>
            </div>
          </div>

          <div className="card-medical p-6">
            <h3 className="font-display font-semibold text-lg text-foreground mb-4">
              Why Weighted FedProx?
            </h3>
            <div className="prose prose-sm max-w-none">
              <p className="text-muted-foreground">
                Our weighted FedProx approach combines two innovations:
              </p>
              <div className="mt-4 space-y-4">
                <div className="p-3 bg-secondary rounded-lg">
                  <p className="font-medium text-foreground">Performance-Based Weighting</p>
                  <p className="text-sm text-muted-foreground">Higher AUROC hospitals have more influence</p>
                  <code className="text-xs mt-2 block">w = 0.6 × AUROC² + 0.4 × (n/Σn)</code>
                </div>
                <div className="p-3 bg-secondary rounded-lg">
                  <p className="font-medium text-foreground">Proximal Regularization</p>
                  <p className="text-sm text-muted-foreground">Prevents model drift from heterogeneous data</p>
                  <code className="text-xs mt-2 block">L = L_local + (μ/2)||θ - θ_global||²</code>
                </div>
              </div>
            </div>
          </div>

          <div className="card-medical p-6 lg:col-span-2">
            <h3 className="font-display font-semibold text-lg text-foreground mb-4">
              Personalization Ensures Specialty Precision
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 bg-secondary rounded-lg">
                <p className="font-medium text-foreground mb-2">Global Knowledge</p>
                <p className="text-sm text-muted-foreground">
                  The federated model learns from all hospitals, providing broad medical knowledge.
                </p>
              </div>
              <div className="p-4 bg-secondary rounded-lg">
                <p className="font-medium text-foreground mb-2">Local Fine-Tuning</p>
                <p className="text-sm text-muted-foreground">
                  Each hospital adds specialty-specific layers for domain expertise.
                </p>
              </div>
              <div className="p-4 bg-primary/10 rounded-lg border border-primary/20">
                <p className="font-medium text-foreground mb-2">Best of Both</p>
                <p className="text-sm text-muted-foreground">
                  Personalized models achieve higher AUROC than either approach alone.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default FederatedLearning;
