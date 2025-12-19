import React from 'react';
import { 
  CheckCircle, 
  AlertCircle, 
  Info, 
  Clock,
  Network,
  Activity,
  Shield,
  Building2
} from 'lucide-react';
import { cn } from '@/lib/utils';

const activities = [
  {
    id: 1,
    type: 'success',
    icon: CheckCircle,
    title: 'FL Round 8 Completed',
    description: 'Global model updated successfully',
    time: '2 mins ago',
  },
  {
    id: 2,
    type: 'info',
    icon: Building2,
    title: 'Hospital E Model Synced',
    description: 'Multi-modal fusion weights applied',
    time: '5 mins ago',
  },
  {
    id: 3,
    type: 'success',
    icon: Activity,
    title: 'Anomaly Detected & Handled',
    description: 'Patient A-2341 arrhythmia alert',
    time: '12 mins ago',
  },
  {
    id: 4,
    type: 'warning',
    icon: AlertCircle,
    title: 'High Drift Detected',
    description: 'Hospital D update deviation: 0.042',
    time: '18 mins ago',
  },
  {
    id: 5,
    type: 'info',
    icon: Shield,
    title: 'DP Noise Applied',
    description: 'Differential privacy ε=1.2 maintained',
    time: '25 mins ago',
  },
  {
    id: 6,
    type: 'success',
    icon: Network,
    title: 'FedProx Regularization',
    description: 'Proximal term μ=0.01 stabilizing updates',
    time: '32 mins ago',
  },
];

export const RecentActivity: React.FC = () => {
  const typeStyles = {
    success: 'bg-success/10 text-success',
    warning: 'bg-warning/10 text-warning',
    error: 'bg-destructive/10 text-destructive',
    info: 'bg-primary/10 text-primary',
  };

  return (
    <div className="card-medical p-6 h-full">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-display font-semibold text-lg text-foreground">Recent Activity</h3>
        <button className="text-sm text-primary hover:underline">View All</button>
      </div>

      <div className="space-y-4 max-h-[400px] overflow-y-auto custom-scrollbar pr-2">
        {activities.map((activity) => (
          <div key={activity.id} className="flex items-start gap-3">
            <div className={cn(
              "w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0",
              typeStyles[activity.type as keyof typeof typeStyles]
            )}>
              <activity.icon className="w-4 h-4" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-foreground">{activity.title}</p>
              <p className="text-xs text-muted-foreground truncate">{activity.description}</p>
            </div>
            <div className="flex items-center gap-1 text-xs text-muted-foreground flex-shrink-0">
              <Clock className="w-3 h-3" />
              <span>{activity.time}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
