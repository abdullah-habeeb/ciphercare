import React from 'react';
import { 
  Network, 
  Activity, 
  Database, 
  Shield,
  TrendingUp,
  Building2
} from 'lucide-react';
import { MetricCard } from '@/components/ui/metric-card';

export const GlobalMetricsPanel: React.FC = () => {
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
      <MetricCard
        title="Global AUROC"
        value="0.942"
        subtitle="Model Performance"
        icon={TrendingUp}
        trend="up"
        trendValue="+3.2%"
        variant="primary"
      />
      <MetricCard
        title="FL Round"
        value="8/10"
        subtitle="Training Progress"
        icon={Network}
        trend="up"
        trendValue="Active"
        variant="success"
      />
      <MetricCard
        title="Active Hospitals"
        value="5"
        subtitle="Connected"
        icon={Building2}
        trend="neutral"
      />
      <MetricCard
        title="Total Samples"
        value="193K"
        subtitle="Training Data"
        icon={Database}
        trend="up"
        trendValue="+12K"
      />
      <MetricCard
        title="IoMT Devices"
        value="247"
        subtitle="Live Connections"
        icon={Activity}
        trend="up"
        trendValue="+8"
        variant="success"
      />
      <MetricCard
        title="DP Status"
        value="Active"
        subtitle="ε = 1.2"
        icon={Shield}
        trend="neutral"
        variant="primary"
      />
    </div>
  );
};
