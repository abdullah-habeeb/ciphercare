import React from 'react';
import { Link } from 'react-router-dom';
import { 
  Network, 
  Activity, 
  Building2, 
  Play,
  Pause,
  RefreshCw,
  Download,
  ChevronRight
} from 'lucide-react';
import { Button } from '@/components/ui/button';

export const QuickActions: React.FC = () => {
  return (
    <div className="card-medical p-6">
      <h3 className="font-display font-semibold text-lg text-foreground mb-4">Quick Actions</h3>
      
      <div className="space-y-3">
        <Link 
          to="/federated-learning"
          className="flex items-center justify-between p-4 bg-secondary rounded-lg hover:bg-secondary/80 transition-colors group"
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
              <Network className="w-5 h-5 text-primary" />
            </div>
            <div>
              <p className="font-medium text-foreground">View FL Pipeline</p>
              <p className="text-sm text-muted-foreground">Training visualization</p>
            </div>
          </div>
          <ChevronRight className="w-5 h-5 text-muted-foreground group-hover:translate-x-1 transition-transform" />
        </Link>

        <Link 
          to="/iomt-monitor"
          className="flex items-center justify-between p-4 bg-secondary rounded-lg hover:bg-secondary/80 transition-colors group"
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-success/10 flex items-center justify-center">
              <Activity className="w-5 h-5 text-success" />
            </div>
            <div>
              <p className="font-medium text-foreground">Live IoMT Stream</p>
              <p className="text-sm text-muted-foreground">Real-time monitoring</p>
            </div>
          </div>
          <ChevronRight className="w-5 h-5 text-muted-foreground group-hover:translate-x-1 transition-transform" />
        </Link>

        <Link 
          to="/hospitals"
          className="flex items-center justify-between p-4 bg-secondary rounded-lg hover:bg-secondary/80 transition-colors group"
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-medical-blue/10 flex items-center justify-center">
              <Building2 className="w-5 h-5 text-medical-blue" />
            </div>
            <div>
              <p className="font-medium text-foreground">Hospital Details</p>
              <p className="text-sm text-muted-foreground">View all institutions</p>
            </div>
          </div>
          <ChevronRight className="w-5 h-5 text-muted-foreground group-hover:translate-x-1 transition-transform" />
        </Link>
      </div>

      <div className="mt-6 pt-4 border-t border-border">
        <p className="text-xs text-muted-foreground uppercase tracking-wider mb-3">FL Controls</p>
        <div className="flex gap-2">
          <Button size="sm" className="flex-1">
            <Play className="w-4 h-4 mr-2" />
            Start Round
          </Button>
          <Button size="sm" variant="outline">
            <Pause className="w-4 h-4" />
          </Button>
          <Button size="sm" variant="outline">
            <RefreshCw className="w-4 h-4" />
          </Button>
          <Button size="sm" variant="outline">
            <Download className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </div>
  );
};
