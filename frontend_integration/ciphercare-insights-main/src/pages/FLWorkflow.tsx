import React, { useState, useEffect } from 'react';
import { 
  Network, 
  Building2, 
  Server, 
  Lock, 
  Shield, 
  ArrowDown, 
  ArrowUp,
  Shuffle,
  Layers,
  CheckCircle,
  Zap,
  RefreshCw,
  Play,
  Pause
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { HOSPITALS } from '@/lib/constants';

const steps = [
  { id: 1, label: 'Local Training', description: 'Each hospital trains on private data', icon: Building2 },
  { id: 2, label: 'DP Noise Added', description: 'Differential privacy protects gradients', icon: Shield },
  { id: 3, label: 'Encrypted Update', description: 'Secure transmission to aggregator', icon: Lock },
  { id: 4, label: 'Weighted Aggregation', description: 'FedProx combines all updates', icon: Server },
  { id: 5, label: 'Global Broadcast', description: 'Updated model sent to all hospitals', icon: Shuffle },
  { id: 6, label: 'Personalization', description: 'Local fine-tuning on specialty data', icon: Layers },
];

const FLWorkflow: React.FC = () => {
  const [activeStep, setActiveStep] = useState(1);
  const [isPlaying, setIsPlaying] = useState(true);
  const [animationPhase, setAnimationPhase] = useState(0);

  useEffect(() => {
    if (!isPlaying) return;
    
    const interval = setInterval(() => {
      setActiveStep(prev => prev >= 6 ? 1 : prev + 1);
      setAnimationPhase(prev => (prev + 1) % 100);
    }, 2000);
    
    return () => clearInterval(interval);
  }, [isPlaying]);

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div className="text-center max-w-3xl mx-auto">
        <h1 className="text-3xl md:text-4xl font-display font-bold text-foreground mb-4 flex items-center justify-center gap-3">
          <Network className="w-10 h-10 text-primary" />
          Federated Learning Workflow
        </h1>
        <p className="text-lg text-muted-foreground">
          Understand how CipherCare enables collaborative AI while preserving data privacy
        </p>
      </div>

      {/* Controls */}
      <div className="flex justify-center gap-4">
        <Button 
          variant="outline" 
          onClick={() => setIsPlaying(!isPlaying)}
          className="gap-2"
        >
          {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          {isPlaying ? 'Pause Animation' : 'Start Animation'}
        </Button>
        <Button 
          variant="outline" 
          onClick={() => setActiveStep(1)}
          className="gap-2"
        >
          <RefreshCw className="w-4 h-4" />
          Reset
        </Button>
      </div>

      {/* Main Visualization */}
      <div className="card-medical p-8">
        {/* Hospital Row */}
        <div className="grid grid-cols-5 gap-4 mb-8">
          {HOSPITALS.map((hospital, i) => (
            <div 
              key={hospital.id}
              className={cn(
                "relative p-4 rounded-2xl border-2 transition-all duration-500",
                activeStep === 1 && "border-primary bg-primary/10 shadow-glow animate-pulse",
                activeStep === 6 && "border-success bg-success/10",
                activeStep !== 1 && activeStep !== 6 && "border-border bg-card"
              )}
            >
              {/* Glowing effect during training */}
              {activeStep === 1 && (
                <div className="absolute inset-0 rounded-2xl bg-primary/20 animate-ping" style={{ animationDuration: '2s' }} />
              )}
              
              <div className="relative">
                <Building2 className={cn(
                  "w-10 h-10 mx-auto mb-2 transition-colors",
                  activeStep === 1 && "text-primary",
                  activeStep === 6 && "text-success",
                  activeStep !== 1 && activeStep !== 6 && "text-muted-foreground"
                )} />
                <p className="text-center font-semibold text-sm text-foreground">{hospital.name}</p>
                <p className="text-center text-xs text-muted-foreground">{hospital.specialty}</p>
                
                {activeStep === 1 && (
                  <div className="mt-2 text-center">
                    <span className="text-xs text-primary font-mono animate-pulse">Training...</span>
                  </div>
                )}
                {activeStep === 6 && (
                  <div className="mt-2 flex justify-center">
                    <CheckCircle className="w-4 h-4 text-success" />
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Arrow Down - DP Noise & Encryption */}
        <div className="flex justify-center mb-8">
          <div className={cn(
            "flex flex-col items-center gap-2 p-4 rounded-xl transition-all duration-500",
            (activeStep === 2 || activeStep === 3) && "bg-warning/10 border border-warning/30"
          )}>
            {[1, 2, 3, 4, 5].map((_, i) => (
              <div 
                key={i} 
                className={cn(
                  "w-1 h-4 rounded-full transition-all",
                  activeStep >= 2 && activeStep <= 3 
                    ? "bg-warning animate-bounce" 
                    : "bg-border"
                )}
                style={{ animationDelay: `${i * 100}ms` }}
              />
            ))}
            <div className="flex items-center gap-2 mt-2">
              {activeStep === 2 && (
                <>
                  <Shield className="w-5 h-5 text-warning" />
                  <span className="text-sm text-warning font-medium">Adding DP Noise (ε=4.5)</span>
                </>
              )}
              {activeStep === 3 && (
                <>
                  <Lock className="w-5 h-5 text-warning" />
                  <span className="text-sm text-warning font-medium">Encrypting Updates</span>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Central Server */}
        <div className="flex justify-center mb-8">
          <div className={cn(
            "relative p-8 rounded-3xl transition-all duration-500",
            activeStep === 4 
              ? "bg-gradient-to-br from-primary to-accent shadow-glow scale-110" 
              : "bg-gradient-to-br from-primary/80 to-accent/80"
          )}>
            {activeStep === 4 && (
              <div className="absolute inset-0 rounded-3xl bg-primary/30 animate-ping" />
            )}
            <div className="relative">
              <Server className="w-16 h-16 text-primary-foreground mx-auto mb-3" />
              <p className="text-center font-bold text-lg text-primary-foreground">Central Aggregator</p>
              <p className="text-center text-sm text-primary-foreground/80">Weighted FedProx</p>
              {activeStep === 4 && (
                <div className="mt-3 text-center">
                  <span className="px-3 py-1 bg-primary-foreground/20 rounded-full text-xs text-primary-foreground font-mono">
                    Aggregating...
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Arrow Up - Broadcast */}
        <div className="flex justify-center mb-8">
          <div className={cn(
            "flex flex-col items-center gap-2 p-4 rounded-xl transition-all duration-500",
            activeStep === 5 && "bg-success/10 border border-success/30"
          )}>
            {activeStep === 5 && (
              <div className="flex items-center gap-2 mb-2">
                <Shuffle className="w-5 h-5 text-success animate-spin" style={{ animationDuration: '3s' }} />
                <span className="text-sm text-success font-medium">Broadcasting Global Model</span>
              </div>
            )}
            {[1, 2, 3, 4, 5].map((_, i) => (
              <div 
                key={i} 
                className={cn(
                  "w-1 h-4 rounded-full transition-all",
                  activeStep === 5 
                    ? "bg-success animate-bounce" 
                    : "bg-border"
                )}
                style={{ animationDelay: `${i * 100}ms`, animationDirection: 'reverse' }}
              />
            ))}
          </div>
        </div>

        {/* Formula */}
        <div className="flex justify-center mb-6">
          <div className="p-4 bg-secondary rounded-xl border border-border">
            <p className="font-mono text-sm text-foreground text-center">
              θ<sub>global</sub> = Σ w<sub>i</sub> · θ<sub>i</sub> + μ/2 ||θ - θ<sub>global</sub>||²
            </p>
            <p className="text-xs text-muted-foreground text-center mt-2">
              Weighted FedProx aggregation with proximal regularization
            </p>
          </div>
        </div>
      </div>

      {/* Step Indicators */}
      <div className="grid grid-cols-6 gap-4">
        {steps.map((step) => {
          const Icon = step.icon;
          const isActive = step.id === activeStep;
          const isComplete = step.id < activeStep;
          
          return (
            <button
              key={step.id}
              onClick={() => setActiveStep(step.id)}
              className={cn(
                "p-4 rounded-xl border-2 text-left transition-all",
                isActive && "border-primary bg-primary/10 shadow-lg",
                isComplete && "border-success/50 bg-success/5",
                !isActive && !isComplete && "border-border bg-card hover:border-primary/30"
              )}
            >
              <div className="flex items-center gap-2 mb-2">
                <div className={cn(
                  "w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold",
                  isActive && "bg-primary text-primary-foreground",
                  isComplete && "bg-success text-success-foreground",
                  !isActive && !isComplete && "bg-muted text-muted-foreground"
                )}>
                  {isComplete ? <CheckCircle className="w-4 h-4" /> : step.id}
                </div>
                <Icon className={cn(
                  "w-4 h-4",
                  isActive && "text-primary",
                  isComplete && "text-success",
                  !isActive && !isComplete && "text-muted-foreground"
                )} />
              </div>
              <p className={cn(
                "font-semibold text-sm",
                isActive && "text-primary",
                isComplete && "text-success",
                !isActive && !isComplete && "text-foreground"
              )}>{step.label}</p>
              <p className="text-xs text-muted-foreground mt-1">{step.description}</p>
            </button>
          );
        })}
      </div>

      {/* Key Benefits */}
      <div className="grid grid-cols-3 gap-6 mt-8">
        <div className="card-medical p-6 text-center">
          <Shield className="w-12 h-12 text-primary mx-auto mb-4" />
          <h3 className="font-display font-semibold text-foreground mb-2">100% Data Privacy</h3>
          <p className="text-sm text-muted-foreground">Raw patient data never leaves the hospital</p>
        </div>
        <div className="card-medical p-6 text-center">
          <Zap className="w-12 h-12 text-success mx-auto mb-4" />
          <h3 className="font-display font-semibold text-foreground mb-2">20%+ AUROC Improvement</h3>
          <p className="text-sm text-muted-foreground">Collaborative learning beats isolated training</p>
        </div>
        <div className="card-medical p-6 text-center">
          <Layers className="w-12 h-12 text-accent mx-auto mb-4" />
          <h3 className="font-display font-semibold text-foreground mb-2">Personalized Models</h3>
          <p className="text-sm text-muted-foreground">Each hospital gets specialty-optimized AI</p>
        </div>
      </div>
    </div>
  );
};

export default FLWorkflow;
