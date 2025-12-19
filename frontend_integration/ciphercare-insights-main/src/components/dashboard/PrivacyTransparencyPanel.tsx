import React, { useState, useEffect } from 'react';
import { Shield, Lock, Eye, EyeOff, Info, AlertTriangle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';

export const PrivacyTransparencyPanel: React.FC = () => {
  const [noiseLevel, setNoiseLevel] = useState(0);
  
  useEffect(() => {
    const interval = setInterval(() => {
      setNoiseLevel(prev => (prev + 1) % 100);
    }, 50);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="card-medical p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
            <Shield className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h3 className="font-display font-semibold text-lg text-foreground">Privacy Transparency</h3>
            <p className="text-sm text-muted-foreground">Differential Privacy Status</p>
          </div>
        </div>
        <Tooltip>
          <TooltipTrigger>
            <Info className="w-5 h-5 text-muted-foreground hover:text-primary cursor-help" />
          </TooltipTrigger>
          <TooltipContent className="max-w-xs">
            <p>Individual patient data cannot be reconstructed due to calibrated DP noise added to all gradient updates.</p>
          </TooltipContent>
        </Tooltip>
      </div>

      {/* DP Parameters */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="p-4 bg-gradient-to-br from-primary/5 to-primary/10 rounded-xl border border-primary/20">
          <div className="flex items-center gap-2 mb-2">
            <Lock className="w-4 h-4 text-primary" />
            <span className="text-sm text-muted-foreground">Epsilon (ε)</span>
          </div>
          <p className="text-3xl font-bold font-mono text-primary">4.5</p>
          <p className="text-xs text-muted-foreground mt-1">Privacy budget</p>
        </div>
        <div className="p-4 bg-gradient-to-br from-success/5 to-success/10 rounded-xl border border-success/20">
          <div className="flex items-center gap-2 mb-2">
            <Shield className="w-4 h-4 text-success" />
            <span className="text-sm text-muted-foreground">Delta (δ)</span>
          </div>
          <p className="text-3xl font-bold font-mono text-success">1e-5</p>
          <p className="text-xs text-muted-foreground mt-1">Failure probability</p>
        </div>
      </div>

      {/* Animated Noise Visualization */}
      <div className="relative p-4 bg-secondary rounded-xl overflow-hidden">
        <div className="flex items-center gap-2 mb-3">
          <Eye className="w-4 h-4 text-muted-foreground" />
          <span className="text-sm font-medium text-foreground">Live DP Noise Application</span>
        </div>
        
        {/* Noise animation bars */}
        <div className="flex gap-1 h-12">
          {Array.from({ length: 30 }).map((_, i) => (
            <div
              key={i}
              className="flex-1 bg-primary/30 rounded-sm transition-all duration-75"
              style={{
                height: `${Math.abs(Math.sin((noiseLevel + i * 12) * 0.1) * 100)}%`,
                opacity: 0.3 + Math.abs(Math.sin((noiseLevel + i * 8) * 0.08) * 0.7),
              }}
            />
          ))}
        </div>
        
        {/* Blur overlay effect */}
        <div className="absolute inset-0 backdrop-blur-[1px] pointer-events-none" 
          style={{ opacity: 0.1 + Math.sin(noiseLevel * 0.05) * 0.1 }} 
        />
      </div>

      {/* Privacy guarantee badge */}
      <div className="mt-4 p-3 bg-success/10 rounded-lg border border-success/20 flex items-center gap-3">
        <div className="w-8 h-8 rounded-full bg-success/20 flex items-center justify-center">
          <Lock className="w-4 h-4 text-success" />
        </div>
        <div className="flex-1">
          <p className="text-sm font-medium text-foreground">Mathematically Guaranteed Privacy</p>
          <p className="text-xs text-muted-foreground">Gradient updates are indistinguishable with/without any individual</p>
        </div>
      </div>
    </div>
  );
};
