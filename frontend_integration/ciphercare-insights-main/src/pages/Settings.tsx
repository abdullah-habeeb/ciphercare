import React from 'react';
import { Settings as SettingsIcon, Shield, Layers, Server, Info, ToggleLeft } from 'lucide-react';
import { Switch } from '@/components/ui/switch';

const Settings: React.FC = () => {
  return (
    <div className="space-y-6 animate-fade-in max-w-3xl">
      <div>
        <h1 className="text-2xl md:text-3xl font-display font-bold text-foreground flex items-center gap-3">
          <SettingsIcon className="w-8 h-8 text-primary" />
          Settings
        </h1>
        <p className="text-muted-foreground">Configure platform preferences</p>
      </div>

      <div className="card-medical p-6 space-y-6">
        <h3 className="font-display font-semibold text-lg text-foreground">Privacy & Security</h3>
        <div className="flex items-center justify-between py-3 border-b border-border">
          <div className="flex items-center gap-3">
            <Shield className="w-5 h-5 text-primary" />
            <div>
              <p className="font-medium text-foreground">Differential Privacy</p>
              <p className="text-sm text-muted-foreground">Add noise to protect individual records (ε = 1.2)</p>
            </div>
          </div>
          <Switch defaultChecked />
        </div>
        <div className="flex items-center justify-between py-3 border-b border-border">
          <div className="flex items-center gap-3">
            <Layers className="w-5 h-5 text-primary" />
            <div>
              <p className="font-medium text-foreground">Personalization Layers</p>
              <p className="text-sm text-muted-foreground">Enable hospital-specific fine-tuning</p>
            </div>
          </div>
          <Switch defaultChecked />
        </div>
      </div>

      <div className="card-medical p-6 space-y-4">
        <h3 className="font-display font-semibold text-lg text-foreground">System Information</h3>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div className="p-3 bg-secondary rounded-lg">
            <p className="text-muted-foreground">Version</p>
            <p className="font-mono text-foreground">CipherCare v2.4.1</p>
          </div>
          <div className="p-3 bg-secondary rounded-lg">
            <p className="text-muted-foreground">Last FL Round</p>
            <p className="font-mono text-foreground">Round 8 — 2 mins ago</p>
          </div>
          <div className="p-3 bg-secondary rounded-lg">
            <p className="text-muted-foreground">FL Engine</p>
            <p className="font-mono text-success">Operational</p>
          </div>
          <div className="p-3 bg-secondary rounded-lg">
            <p className="text-muted-foreground">IoMT Gateway</p>
            <p className="font-mono text-success">Operational</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Settings;
