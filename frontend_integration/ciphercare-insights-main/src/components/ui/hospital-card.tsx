import React from 'react';
import { Link } from 'react-router-dom';
import { cn } from '@/lib/utils';
import { 
  Heart, 
  Activity, 
  Stethoscope, 
  Users, 
  Building2,
  ChevronRight,
  Clock,
  Database
} from 'lucide-react';
import { Button } from './button';

const iconMap: Record<string, React.ElementType> = {
  Heart,
  Activity,
  Stethoscope,
  Users,
  Building2,
};

interface HospitalCardProps {
  hospital: {
    id: string;
    name: string;
    fullName: string;
    specialty: string;
    icon: string;
    color: string;
    samples: number;
    localAuroc: number;
    contributionWeight: number;
    lastUpdated: string;
    status: string;
  };
  className?: string;
}

export const HospitalCard: React.FC<HospitalCardProps> = ({ hospital, className }) => {
  const Icon = iconMap[hospital.icon] || Building2;

  const colorClasses: Record<string, string> = {
    'medical-red': 'from-medical-red/20 to-medical-red/5 border-medical-red/30',
    'medical-orange': 'from-medical-orange/20 to-medical-orange/5 border-medical-orange/30',
    'medical-blue': 'from-medical-blue/20 to-medical-blue/5 border-medical-blue/30',
    'medical-purple': 'from-medical-purple/20 to-medical-purple/5 border-medical-purple/30',
    'medical-teal': 'from-primary/20 to-primary/5 border-primary/30',
  };

  const iconColorClasses: Record<string, string> = {
    'medical-red': 'bg-medical-red/20 text-medical-red',
    'medical-orange': 'bg-medical-orange/20 text-medical-orange',
    'medical-blue': 'bg-medical-blue/20 text-medical-blue',
    'medical-purple': 'bg-medical-purple/20 text-medical-purple',
    'medical-teal': 'bg-primary/20 text-primary',
  };

  return (
    <div className={cn(
      "card-medical-glow bg-gradient-to-br border p-6 group",
      colorClasses[hospital.color],
      className
    )}>
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className={cn(
            "w-12 h-12 rounded-xl flex items-center justify-center",
            iconColorClasses[hospital.color]
          )}>
            <Icon className="w-6 h-6" />
          </div>
          <div>
            <h3 className="font-display font-semibold text-foreground">{hospital.name}</h3>
            <p className="text-sm text-muted-foreground">{hospital.specialty}</p>
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          <div className={cn(
            "status-dot",
            hospital.status === 'online' ? "status-online status-dot-pulse" : "status-warning"
          )} />
          <span className="text-xs text-muted-foreground capitalize">{hospital.status}</span>
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Local AUROC</p>
          <p className="text-2xl font-bold font-mono text-foreground">{hospital.localAuroc.toFixed(3)}</p>
        </div>
        <div>
          <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">FL Weight</p>
          <p className="text-2xl font-bold font-mono text-foreground">{(hospital.contributionWeight * 100).toFixed(0)}%</p>
        </div>
      </div>

      {/* Footer Info */}
      <div className="flex items-center gap-4 mb-4 text-sm text-muted-foreground">
        <div className="flex items-center gap-1.5">
          <Database className="w-4 h-4" />
          <span>{hospital.samples.toLocaleString()}</span>
        </div>
        <div className="flex items-center gap-1.5">
          <Clock className="w-4 h-4" />
          <span>{hospital.lastUpdated}</span>
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-2">
        <Button asChild variant="secondary" className="flex-1 group-hover:bg-primary group-hover:text-primary-foreground transition-all">
          <Link to={`/hospitals/${hospital.id}`}>
            View Details
            <ChevronRight className="w-4 h-4 ml-1 transition-transform group-hover:translate-x-1" />
          </Link>
        </Button>
      </div>
    </div>
  );
};
