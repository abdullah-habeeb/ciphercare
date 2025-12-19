import React from 'react';
import { cn } from '@/lib/utils';
import { LucideIcon } from 'lucide-react';

interface StatusBadgeProps {
  status: 'online' | 'offline' | 'warning' | 'processing';
  label?: string;
  icon?: LucideIcon;
  className?: string;
  pulse?: boolean;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  label,
  icon: Icon,
  className,
  pulse = false,
}) => {
  const statusStyles = {
    online: 'bg-success/10 text-success border-success/20',
    offline: 'bg-destructive/10 text-destructive border-destructive/20',
    warning: 'bg-warning/10 text-warning border-warning/20',
    processing: 'bg-primary/10 text-primary border-primary/20',
  };

  const dotStyles = {
    online: 'bg-success',
    offline: 'bg-destructive',
    warning: 'bg-warning',
    processing: 'bg-primary',
  };

  return (
    <span className={cn(
      "inline-flex items-center gap-2 px-3 py-1.5 rounded-full border text-sm font-medium",
      statusStyles[status],
      className
    )}>
      <span className={cn(
        "w-2 h-2 rounded-full",
        dotStyles[status],
        pulse && "animate-pulse"
      )} />
      {Icon && <Icon className="w-3.5 h-3.5" />}
      {label && <span>{label}</span>}
    </span>
  );
};
