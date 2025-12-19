import React from 'react';
import { HOSPITALS } from '@/lib/constants';
import { HospitalCard } from '@/components/ui/hospital-card';

export const HospitalOverview: React.FC = () => {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="font-display font-semibold text-lg text-foreground">Hospital Network</h3>
          <p className="text-sm text-muted-foreground">Connected institutions in the federation</p>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4 stagger-animation">
        {HOSPITALS.map((hospital) => (
          <HospitalCard key={hospital.id} hospital={hospital} />
        ))}
      </div>
    </div>
  );
};
