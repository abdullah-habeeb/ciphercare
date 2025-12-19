import React from 'react';
import { Heart } from 'lucide-react';
import { cn } from '@/lib/utils';

interface PulseIndicatorProps {
  heartRate: number;
  className?: string;
}

export const PulseIndicator: React.FC<PulseIndicatorProps> = ({ heartRate, className }) => {
  // Calculate pulse interval in milliseconds
  const pulseInterval = (60 / heartRate) * 1000;
  
  return (
    <div className={cn("relative flex items-center justify-center", className)}>
      <div className="relative">
        {/* Outer pulse ring */}
        <div
          className="absolute inset-0 rounded-full bg-success/30"
          style={{
            animation: `pulse-ring ${pulseInterval}ms cubic-bezier(0.4, 0, 0.6, 1) infinite`,
          }}
        />
        {/* Inner pulse ring */}
        <div
          className="absolute inset-0 rounded-full bg-success/20"
          style={{
            animation: `pulse-ring ${pulseInterval}ms cubic-bezier(0.4, 0, 0.6, 1) infinite`,
            animationDelay: `${pulseInterval / 2}ms`,
          }}
        />
        {/* Heart icon */}
        <div className="relative w-12 h-12 flex items-center justify-center">
          <Heart
            className="w-6 h-6 text-success"
            style={{
              animation: `pulse-soft ${pulseInterval}ms ease-in-out infinite`,
            }}
            fill="currentColor"
          />
        </div>
      </div>
    </div>
  );
};





