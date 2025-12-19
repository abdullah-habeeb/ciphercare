import React from 'react';
import { User, Calendar, MapPin, Stethoscope, UserCircle } from 'lucide-react';
import { PatientInfo } from '@/lib/iomt-simulator';
import { cn } from '@/lib/utils';

interface PatientInfoCardProps {
  patient: PatientInfo;
  className?: string;
}

export const PatientInfoCard: React.FC<PatientInfoCardProps> = ({ patient, className }) => {
  return (
    <div className={cn("card-medical p-6", className)}>
      <div className="flex items-start gap-4">
        <div className="w-16 h-16 rounded-full bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center flex-shrink-0">
          <UserCircle className="w-8 h-8 text-primary" />
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="text-lg font-display font-bold text-foreground mb-1">
            {patient.name}
          </h3>
          <div className="grid grid-cols-2 gap-3 mt-3">
            <div className="flex items-center gap-2 text-sm">
              <User className="w-4 h-4 text-muted-foreground" />
              <span className="text-muted-foreground">Age:</span>
              <span className="font-medium text-foreground">{patient.age} {patient.gender}</span>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <Calendar className="w-4 h-4 text-muted-foreground" />
              <span className="text-muted-foreground">ID:</span>
              <span className="font-mono font-medium text-foreground">{patient.id}</span>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <MapPin className="w-4 h-4 text-muted-foreground" />
              <span className="text-muted-foreground">Ward:</span>
              <span className="font-medium text-foreground">{patient.ward}</span>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <MapPin className="w-4 h-4 text-muted-foreground" />
              <span className="text-muted-foreground">Room:</span>
              <span className="font-medium text-foreground">{patient.room}</span>
            </div>
          </div>
          <div className="mt-4 pt-4 border-t border-border">
            <div className="flex items-start gap-2 text-sm">
              <Stethoscope className="w-4 h-4 text-muted-foreground mt-0.5" />
              <div>
                <span className="text-muted-foreground">Diagnosis: </span>
                <span className="font-medium text-foreground">{patient.diagnosis}</span>
              </div>
            </div>
            <div className="flex items-center gap-2 text-sm mt-2">
              <User className="w-4 h-4 text-muted-foreground" />
              <span className="text-muted-foreground">Attending: </span>
              <span className="font-medium text-foreground">{patient.attendingPhysician}</span>
            </div>
            <div className="mt-3 flex items-center gap-2">
              <span className="text-xs text-muted-foreground">Device:</span>
              <span className="text-xs font-mono px-2 py-1 bg-primary/10 text-primary rounded">
                {patient.deviceId}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};





