import React from 'react';
import { Link } from 'react-router-dom';
import { 
  Shield, 
  Lock, 
  Network, 
  Zap, 
  Building2, 
  Server, 
  TrendingUp,
  CheckCircle,
  ArrowRight,
  Play,
  Star,
  Heart,
  Brain,
  Globe
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { HOSPITALS } from '@/lib/constants';

const ProductStory: React.FC = () => {
  return (
    <div className="space-y-16 animate-fade-in pb-12">
      {/* Hero Section */}
      <section className="text-center max-w-4xl mx-auto pt-8">
        <div className="inline-flex items-center gap-2 px-4 py-2 bg-primary/10 rounded-full text-primary text-sm font-medium mb-6">
          <Shield className="w-4 h-4" />
          Privacy-First Healthcare AI
        </div>
        <h1 className="text-4xl md:text-6xl font-display font-bold text-foreground mb-6">
          CipherCare
        </h1>
        <p className="text-xl md:text-2xl text-muted-foreground mb-8 leading-relaxed">
          The world's first <span className="text-primary font-semibold">federated medical AI platform</span> that enables 
          hospitals to collaborate on AI without sharing patient data.
        </p>
        <div className="flex justify-center gap-4">
          <Button asChild size="lg" className="gap-2">
            <Link to="/">
              <Play className="w-5 h-5" />
              View Live Demo
            </Link>
          </Button>
          <Button asChild variant="outline" size="lg" className="gap-2">
            <Link to="/fl-workflow">
              See How It Works
            </Link>
          </Button>
        </div>
      </section>

      {/* Problem Section */}
      <section className="max-w-5xl mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-display font-bold text-foreground mb-4">The Problem</h2>
          <p className="text-lg text-muted-foreground">Healthcare AI is broken by data silos</p>
        </div>
        <div className="grid grid-cols-3 gap-6">
          <div className="card-medical p-6 border-destructive/30 bg-destructive/5">
            <Lock className="w-10 h-10 text-destructive mb-4" />
            <h3 className="font-semibold text-foreground mb-2">Data Privacy Laws</h3>
            <p className="text-sm text-muted-foreground">
              HIPAA, GDPR, and local regulations prevent hospitals from sharing patient data
            </p>
          </div>
          <div className="card-medical p-6 border-destructive/30 bg-destructive/5">
            <Building2 className="w-10 h-10 text-destructive mb-4" />
            <h3 className="font-semibold text-foreground mb-2">Isolated Data Silos</h3>
            <p className="text-sm text-muted-foreground">
              Each hospital trains on limited local data, resulting in poor generalization
            </p>
          </div>
          <div className="card-medical p-6 border-destructive/30 bg-destructive/5">
            <TrendingUp className="w-10 h-10 text-destructive mb-4" />
            <h3 className="font-semibold text-foreground mb-2">Suboptimal Models</h3>
            <p className="text-sm text-muted-foreground">
              AI models underperform because they lack diverse training data
            </p>
          </div>
        </div>
      </section>

      {/* Solution Section */}
      <section className="max-w-5xl mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-display font-bold text-foreground mb-4">Our Solution</h2>
          <p className="text-lg text-muted-foreground">Federated Learning + Differential Privacy + Personalization</p>
        </div>
        
        {/* Visual Diagram */}
        <div className="card-medical p-8 mb-8">
          <div className="grid grid-cols-5 gap-4 mb-8">
            {HOSPITALS.map((h, i) => (
              <div key={h.id} className="text-center">
                <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center mx-auto mb-2">
                  <Building2 className="w-8 h-8 text-primary" />
                </div>
                <p className="text-sm font-medium text-foreground">{h.name}</p>
                <p className="text-xs text-muted-foreground">{h.specialty}</p>
              </div>
            ))}
          </div>
          
          <div className="flex justify-center mb-6">
            <div className="flex items-center gap-4">
              <ArrowRight className="w-6 h-6 text-primary rotate-90" />
              <span className="text-sm text-muted-foreground">Encrypted Gradients Only</span>
              <ArrowRight className="w-6 h-6 text-primary rotate-90" />
            </div>
          </div>
          
          <div className="flex justify-center mb-6">
            <div className="w-24 h-24 rounded-3xl bg-gradient-to-br from-primary to-accent flex items-center justify-center shadow-glow">
              <Server className="w-12 h-12 text-primary-foreground" />
            </div>
          </div>
          
          <p className="text-center text-muted-foreground">
            Central aggregator combines model updates without ever seeing raw data
          </p>
        </div>

        {/* Key Innovations */}
        <div className="grid grid-cols-3 gap-6">
          <div className="card-medical p-6 border-primary/30 bg-primary/5">
            <Network className="w-10 h-10 text-primary mb-4" />
            <h3 className="font-semibold text-foreground mb-2">Weighted FedProx</h3>
            <p className="text-sm text-muted-foreground mb-3">
              Higher-quality hospitals contribute more to the global model
            </p>
            <code className="text-xs bg-secondary px-2 py-1 rounded block">
              w = 0.6×AUROC² + 0.4×(n/Σn)
            </code>
          </div>
          <div className="card-medical p-6 border-success/30 bg-success/5">
            <Shield className="w-10 h-10 text-success mb-4" />
            <h3 className="font-semibold text-foreground mb-2">Differential Privacy</h3>
            <p className="text-sm text-muted-foreground mb-3">
              Mathematically guaranteed protection for individual patients
            </p>
            <code className="text-xs bg-secondary px-2 py-1 rounded block">
              ε = 4.5, δ = 1e-5
            </code>
          </div>
          <div className="card-medical p-6 border-accent/30 bg-accent/5">
            <Brain className="w-10 h-10 text-accent mb-4" />
            <h3 className="font-semibold text-foreground mb-2">Personalization</h3>
            <p className="text-sm text-muted-foreground mb-3">
              Each hospital fine-tunes the global model for their specialty
            </p>
            <code className="text-xs bg-secondary px-2 py-1 rounded block">
              +1.5-3.2% AUROC boost
            </code>
          </div>
        </div>
      </section>

      {/* Results Section */}
      <section className="max-w-5xl mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-display font-bold text-foreground mb-4">Proven Results</h2>
          <p className="text-lg text-muted-foreground">Before vs After Federated Learning</p>
        </div>
        
        <div className="grid grid-cols-2 gap-8 mb-8">
          <div className="card-medical p-8 text-center border-muted bg-muted/30">
            <p className="text-sm text-muted-foreground uppercase tracking-wider mb-4">Before FL</p>
            <p className="text-6xl font-bold font-mono text-muted-foreground">0.782</p>
            <p className="text-muted-foreground mt-2">Average Hospital AUROC</p>
            <div className="mt-4 flex justify-center gap-2">
              {[1,2,3].map(i => <Star key={i} className="w-5 h-5 text-muted" />)}
              {[4,5].map(i => <Star key={i} className="w-5 h-5 text-muted/30" />)}
            </div>
          </div>
          <div className="card-medical p-8 text-center border-success/50 bg-gradient-to-br from-success/5 to-success/10">
            <p className="text-sm text-success uppercase tracking-wider mb-4">After FL</p>
            <p className="text-6xl font-bold font-mono text-success">0.942</p>
            <p className="text-foreground mt-2">Global Model AUROC</p>
            <div className="mt-4 flex justify-center gap-2">
              {[1,2,3,4,5].map(i => <Star key={i} className="w-5 h-5 text-success fill-success" />)}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-4 gap-4">
          <div className="p-4 bg-secondary rounded-xl text-center">
            <p className="text-3xl font-bold text-primary">+20.5%</p>
            <p className="text-sm text-muted-foreground">AUROC Improvement</p>
          </div>
          <div className="p-4 bg-secondary rounded-xl text-center">
            <p className="text-3xl font-bold text-foreground">193K</p>
            <p className="text-sm text-muted-foreground">Training Samples</p>
          </div>
          <div className="p-4 bg-secondary rounded-xl text-center">
            <p className="text-3xl font-bold text-foreground">5</p>
            <p className="text-sm text-muted-foreground">Hospitals</p>
          </div>
          <div className="p-4 bg-secondary rounded-xl text-center">
            <p className="text-3xl font-bold text-foreground">0</p>
            <p className="text-sm text-muted-foreground">Data Shared</p>
          </div>
        </div>
      </section>

      {/* Why Multi-Specialty Matters */}
      <section className="max-w-5xl mx-auto">
        <div className="card-medical p-8 bg-gradient-to-br from-primary/5 to-accent/5 border-primary/30">
          <div className="flex items-start gap-6">
            <div className="w-16 h-16 rounded-2xl bg-primary/20 flex items-center justify-center flex-shrink-0">
              <Globe className="w-8 h-8 text-primary" />
            </div>
            <div>
              <h3 className="text-xl font-display font-bold text-foreground mb-2">
                Why Hospital E Validates Our Approach
              </h3>
              <p className="text-muted-foreground mb-4">
                Hospital E is our multi-specialty center that combines ECG, X-ray, and vitals data. 
                Its exceptional performance (0.962 AUROC) proves that federated learning truly 
                enables cross-domain knowledge transfer.
              </p>
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-success" />
                  <span className="text-sm text-foreground">Multi-modal fusion</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-success" />
                  <span className="text-sm text-foreground">Highest sample count (67K)</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-success" />
                  <span className="text-sm text-foreground">Best AUROC in network</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="text-center max-w-3xl mx-auto">
        <div className="card-medical p-12 bg-gradient-to-br from-primary/10 to-accent/10 border-primary/30">
          <Heart className="w-16 h-16 text-primary mx-auto mb-6" />
          <h2 className="text-3xl font-display font-bold text-foreground mb-4">
            Ready to Transform Healthcare AI?
          </h2>
          <p className="text-lg text-muted-foreground mb-8">
            Join the CipherCare network and unlock the power of collaborative AI 
            while keeping your patient data 100% private.
          </p>
          <div className="flex justify-center gap-4">
            <Button asChild size="lg" className="gap-2">
              <Link to="/">
                <Play className="w-5 h-5" />
                Explore the Platform
              </Link>
            </Button>
            <Button asChild variant="outline" size="lg">
              <Link to="/federated-learning">
                View FL Pipeline
              </Link>
            </Button>
          </div>
        </div>
      </section>
    </div>
  );
};

export default ProductStory;
