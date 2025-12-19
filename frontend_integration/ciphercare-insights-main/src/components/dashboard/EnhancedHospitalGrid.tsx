import React from 'react';
import { Link } from 'react-router-dom';
import { HOSPITALS } from '@/lib/constants';
import { 
  Heart, 
  Activity, 
  Stethoscope, 
  Users, 
  Building2,
  ChevronRight,
  Clock,
  Database,
  Shield,
  Zap
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Progress } from '@/components/ui/progress';

const iconMap: Record<string, React.ElementType> = {
  Heart,
  Activity,
  Stethoscope,
  Users,
  Building2,
};

const colorClasses: Record<string, { bg: string; border: string; text: string }> = {
  'medical-red': { bg: 'from-medical-red/10 to-medical-red/5', border: 'border-medical-red/30', text: 'text-medical-red' },
  'medical-orange': { bg: 'from-medical-orange/10 to-medical-orange/5', border: 'border-medical-orange/30', text: 'text-medical-orange' },
  'medical-blue': { bg: 'from-medical-blue/10 to-medical-blue/5', border: 'border-medical-blue/30', text: 'text-medical-blue' },
  'medical-purple': { bg: 'from-medical-purple/10 to-medical-purple/5', border: 'border-medical-purple/30', text: 'text-medical-purple' },
  'medical-teal': { bg: 'from-primary/10 to-primary/5', border: 'border-primary/30', text: 'text-primary' },
};

export const EnhancedHospitalGrid: React.FC = () => {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="font-display font-semibold text-lg text-foreground">Hospital Network</h3>
          <p className="text-sm text-muted-foreground">5 institutions in the federation</p>
        </div>
        <Link to="/hospitals" className="text-sm text-primary hover:underline flex items-center gap-1">
          View All <ChevronRight className="w-4 h-4" />
        </Link>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-4">
        {HOSPITALS.map((hospital, index) => {
          const Icon = iconMap[hospital.icon] || Building2;
          const colors = colorClasses[hospital.color];
          const epsilon = (2.5 + Math.random() * 2).toFixed(1);
          
          return (
            <Link
              key={hospital.id}
              to={`/hospitals/${hospital.id}`}
              className={cn(
                "group card-medical p-4 bg-gradient-to-br border transition-all duration-300",
                colors.bg,
                colors.border,
                "hover:shadow-lg hover:scale-[1.02]"
              )}
              style={{ animationDelay: `${index * 100}ms` }}
            >
              {/* Header */}
              <div className="flex items-center justify-between mb-3">
                <div className={cn("w-10 h-10 rounded-xl flex items-center justify-center", `bg-${hospital.color}/20`)}>
                  <Icon className={cn("w-5 h-5", colors.text)} />
                </div>
                <div className="flex items-center gap-1">
                  <span className="status-dot status-online status-dot-pulse" />
                </div>
              </div>

              {/* Name & Specialty */}
              <h4 className="font-display font-semibold text-foreground mb-1">{hospital.name}</h4>
              <p className="text-xs text-muted-foreground mb-3 line-clamp-1">{hospital.specialty}</p>

              {/* Metrics Grid */}
              <div className="space-y-2 mb-3">
                <div className="flex justify-between items-center">
                  <span className="text-xs text-muted-foreground">AUROC</span>
                  <span className="text-sm font-bold font-mono text-foreground">{hospital.localAuroc.toFixed(3)}</span>
                </div>
                <Progress value={hospital.localAuroc * 100} className="h-1.5" />
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="flex items-center gap-1 text-muted-foreground">
                  <Database className="w-3 h-3" />
                  <span>{(hospital.samples / 1000).toFixed(0)}K</span>
                </div>
                <div className="flex items-center gap-1 text-muted-foreground">
                  <Shield className="w-3 h-3" />
                  <span>ε={epsilon}</span>
                </div>
                <div className="flex items-center gap-1 text-muted-foreground">
                  <Zap className="w-3 h-3" />
                  <span>w={hospital.contributionWeight.toFixed(2)}</span>
                </div>
                <div className="flex items-center gap-1 text-muted-foreground">
                  <Clock className="w-3 h-3" />
                  <span>{hospital.lastUpdated}</span>
                </div>
              </div>

              {/* Hover indicator */}
              <div className="mt-3 pt-3 border-t border-border/50 flex items-center justify-between opacity-0 group-hover:opacity-100 transition-opacity">
                <span className="text-xs text-primary font-medium">View Details</span>
                <ChevronRight className="w-4 h-4 text-primary transform group-hover:translate-x-1 transition-transform" />
              </div>
            </Link>
          );
        })}
      </div>
    </div>
  );
};
