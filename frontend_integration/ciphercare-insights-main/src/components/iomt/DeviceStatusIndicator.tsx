import React from 'react';
import { Battery, Wifi, Clock, AlertTriangle } from 'lucide-react';
import { DeviceStatus } from '@/lib/iomt-simulator';
import { cn } from '@/lib/utils';
import { Progress } from '@/components/ui/progress';

interface DeviceStatusIndicatorProps {
  device: DeviceStatus;
  className?: string;
}

export const DeviceStatusIndicator: React.FC<DeviceStatusIndicatorProps> = ({ device, className }) => {
  const getSignalBars = (strength: number) => {
    const bars = Math.ceil(strength / 25);
    return Array.from({ length: 4 }, (_, i) => i < bars);
  };

  const getStatusColor = () => {
    if (device.status === 'online') return 'text-success';
    if (device.status === 'warning') return 'text-warning';
    return 'text-destructive';
  };

  const getBatteryColor = () => {
    if (device.battery > 50) return 'text-success';
    if (device.battery > 20) return 'text-warning';
    return 'text-destructive';
  };

  return (
    <div className={cn("card-medical p-4", className)}>
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1 min-w-0">
          <h4 className="font-medium text-foreground text-sm mb-1">{device.name}</h4>
          <p className="text-xs text-muted-foreground font-mono">{device.id}</p>
        </div>
        <div className={cn("flex items-center gap-1", getStatusColor())}>
          <div className={cn("status-dot", device.status === 'online' ? "status-online" : device.status === 'warning' ? "status-warning" : "status-offline")} />
          <span className="text-xs font-medium capitalize">{device.status}</span>
        </div>
      </div>

      <div className="space-y-3">
        {/* Battery */}
        <div>
          <div className="flex items-center justify-between mb-1">
            <div className="flex items-center gap-1.5">
              <Battery className={cn("w-3.5 h-3.5", getBatteryColor())} />
              <span className="text-xs text-muted-foreground">Battery</span>
            </div>
            <span className={cn("text-xs font-mono font-medium", getBatteryColor())}>
              {device.battery}%
            </span>
          </div>
          <Progress value={device.battery} className="h-1.5" />
        </div>

        {/* Signal Strength */}
        <div>
          <div className="flex items-center justify-between mb-1">
            <div className="flex items-center gap-1.5">
              <Wifi className="w-3.5 h-3.5 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">Signal</span>
            </div>
            <div className="flex items-center gap-0.5">
              {getSignalBars(device.signalStrength).map((filled, i) => (
                <div
                  key={i}
                  className={cn(
                    "w-1 h-3 rounded-sm",
                    filled ? "bg-success" : "bg-muted"
                  )}
                  style={{ height: `${(i + 1) * 3 + 2}px` }}
                />
              ))}
            </div>
          </div>
        </div>

        {/* Latency */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <Clock className="w-3.5 h-3.5 text-muted-foreground" />
            <span className="text-xs text-muted-foreground">Latency</span>
          </div>
          <span className={cn(
            "text-xs font-mono",
            device.latency < 20 ? "text-success" : device.latency < 50 ? "text-warning" : "text-destructive"
          )}>
            {device.latency}ms
          </span>
        </div>

        {/* Last Update */}
        <div className="pt-2 border-t border-border">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Last Update</span>
            <span className="text-xs font-mono text-foreground">
              {device.lastUpdate > Date.now() - 60000
                ? 'Just now'
                : `${Math.floor((Date.now() - device.lastUpdate) / 1000)}s ago`}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};





