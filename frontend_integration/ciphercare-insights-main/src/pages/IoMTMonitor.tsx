import React, { useState, useEffect, useRef } from 'react';
import { 
  Activity, 
  Heart, 
  Thermometer, 
  Wind, 
  AlertTriangle, 
  Wifi, 
  WifiOff, 
  Play, 
  Pause,
  ChevronDown,
  ChevronUp,
  Settings,
  Volume2,
  VolumeX
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { GaugeChart } from '@/components/ui/gauge-chart';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { cn } from '@/lib/utils';
import { 
  iomtSimulator, 
  MOCK_PATIENTS, 
  MOCK_DEVICES,
  VitalsData 
} from '@/lib/iomt-simulator';
import { ECGWaveform } from '@/components/iomt/ECGWaveform';
import { PatientInfoCard } from '@/components/iomt/PatientInfoCard';
import { DeviceStatusIndicator } from '@/components/iomt/DeviceStatusIndicator';
import { VitalsTrendChart } from '@/components/iomt/VitalsTrendChart';
import { AnomalySimulator } from '@/components/iomt/AnomalySimulator';
import { PulseIndicator } from '@/components/iomt/PulseIndicator';
import { RiskPredictionPanel } from '@/components/iomt/RiskPredictionPanel';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { predictDeterioration, PredictionResponse } from '@/lib/api';

const IoMTMonitor: React.FC = () => {
  const [isLive, setIsLive] = useState(true);
  const [ecgData, setEcgData] = useState(iomtSimulator.generateECGData(100));
  const [vitals, setVitals] = useState<VitalsData>(iomtSimulator.generateVitals());
  const [vitalsHistory, setVitalsHistory] = useState<VitalsData[]>([]);
  const [anomaly, setAnomaly] = useState<string | null>(null);
  const [showTrends, setShowTrends] = useState(false);
  const [playbackPosition, setPlaybackPosition] = useState(60);
  const [selectedPatient] = useState(MOCK_PATIENTS[0]);
  const [devices] = useState(MOCK_DEVICES);
  
  // ML Prediction state
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [riskHistory, setRiskHistory] = useState<Array<{ timestamp: number; riskScore: number }>>([]);
  const [isMuted, setIsMuted] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Update vitals and ECG in real-time
  useEffect(() => {
    if (!isLive) return;

    const interval = setInterval(() => {
      // Generate new ECG data (rolling buffer)
      const newECG = iomtSimulator.generateECGData(10);
      setEcgData(prev => [...prev.slice(-90), ...newECG]);
      
      // Generate new vitals
      const newVitals = iomtSimulator.generateVitals();
      setVitals(newVitals);
      
      // Update history
      const history = iomtSimulator.getVitalsHistory(60);
      setVitalsHistory(history);
      
      // Check for anomalies
      const currentAnomaly = iomtSimulator.getCurrentAnomaly();
      if (currentAnomaly.type !== 'none') {
        const anomalyLabels: Record<string, string> = {
          arrhythmia: 'Irregular heart rhythm detected',
          tachycardia: 'Tachycardia detected (>120 BPM)',
          bradycardia: 'Bradycardia detected (<60 BPM)',
          hypoxia: 'Low oxygen saturation detected (<90%)',
          fever: 'Elevated temperature detected (>38.5°C)',
        };
        setAnomaly(anomalyLabels[currentAnomaly.type] || 'Anomaly detected');
      } else {
        // Random natural anomalies (rare)
        if (Math.random() > 0.98 && !anomaly) {
          setAnomaly('Minor rhythm variation detected');
          setTimeout(() => setAnomaly(null), 5000);
        } else if (anomaly && Math.random() > 0.7) {
          setAnomaly(null);
        }
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [isLive, anomaly]);

  // ML Prediction API call
  useEffect(() => {
    if (!isLive) return;

    const predictInterval = setInterval(async () => {
      try {
        const predictionResult = await predictDeterioration({
          heartRate: vitals.heartRate,
          spo2: vitals.spo2,
          temperature: vitals.temperature,
          respRate: vitals.respiratoryRate,
          systolicBP: vitals.systolicBP,
          diastolicBP: vitals.diastolicBP,
        });

        setPrediction(predictionResult);

        // Update risk history
        setRiskHistory(prev => {
          const newHistory = [...prev, {
            timestamp: Date.now(),
            riskScore: predictionResult.riskScore,
          }];
          // Keep last 60 points
          return newHistory.slice(-60);
        });

        // Play alert sound for critical cases
        if (predictionResult.severity === 'Critical' && !isMuted) {
          // Create audio context for beep sound
          const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
          const oscillator = audioContext.createOscillator();
          const gainNode = audioContext.createGain();
          
          oscillator.connect(gainNode);
          gainNode.connect(audioContext.destination);
          
          oscillator.frequency.value = 800;
          oscillator.type = 'sine';
          
          gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
          gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
          
          oscillator.start(audioContext.currentTime);
          oscillator.stop(audioContext.currentTime + 0.5);
        }
      } catch (error) {
        console.error('Prediction error:', error);
      }
    }, 1000);

    return () => clearInterval(predictInterval);
  }, [isLive, vitals, isMuted]);

  const handleTriggerAnomaly = (type: 'arrhythmia' | 'tachycardia' | 'bradycardia' | 'hypoxia' | 'fever', intensity: number, duration: number) => {
    iomtSimulator.triggerAnomaly(type, intensity, duration);
    setAnomaly(null); // Clear previous anomaly message
  };

  const handleClearAnomaly = () => {
    iomtSimulator.clearAnomaly();
    setAnomaly(null);
  };

  const currentAnomaly = iomtSimulator.getCurrentAnomaly();

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-2xl md:text-3xl font-display font-bold text-foreground flex items-center gap-3">
            <Activity className="w-8 h-8 text-primary" />
            Live IoMT Monitor
          </h1>
          <p className="text-muted-foreground">Real-time patient vitals streaming • Command Center</p>
        </div>
        <div className="flex items-center gap-3">
          <div className={cn(
            "flex items-center gap-2 px-4 py-2 rounded-lg border transition-all",
            isLive 
              ? "bg-success/10 border-success/20 text-success" 
              : "bg-muted border-border text-muted-foreground"
          )}>
            {isLive ? (
              <>
                <Wifi className="w-4 h-4" />
                <span className="text-sm font-medium">Live Streaming</span>
              </>
            ) : (
              <>
                <WifiOff className="w-4 h-4" />
                <span className="text-sm font-medium">Paused</span>
              </>
            )}
          </div>
          <Button variant="outline" size="sm" onClick={() => setIsLive(!isLive)}>
            {isLive ? <Pause className="w-4 h-4 mr-2" /> : <Play className="w-4 h-4 mr-2" />}
            {isLive ? 'Pause' : 'Resume'}
          </Button>
        </div>
      </div>

      {/* Critical Risk Alert */}
      {prediction && prediction.severity === 'Critical' && (
        <div className={cn(
          "p-4 border-2 rounded-lg flex items-center justify-between animate-fade-in",
          "bg-destructive/20 border-destructive/50",
          prediction.severity === 'Critical' && "animate-pulse"
        )}>
          <div className="flex items-center gap-3">
            <AlertTriangle className="w-6 h-6 text-destructive animate-pulse" />
            <div>
              <span className="font-bold text-destructive text-lg">
                CRITICAL DETERIORATION RISK DETECTED
              </span>
              <p className="text-sm text-destructive/80 mt-0.5">
                Risk Score: {(prediction.riskScore * 100).toFixed(1)}% • {prediction.recommendedAction}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsMuted(!isMuted)}
              className="h-8 w-8"
            >
              {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
            </Button>
          </div>
        </div>
      )}

      {/* Anomaly Alert */}
      {anomaly && (
        <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-lg flex items-center justify-between animate-fade-in">
          <div className="flex items-center gap-3">
            <AlertTriangle className="w-5 h-5 text-destructive animate-pulse" />
            <div>
          <span className="font-medium text-destructive">{anomaly}</span>
              <p className="text-sm text-muted-foreground mt-0.5">
                Patient {selectedPatient.id} • {selectedPatient.ward}
              </p>
            </div>
          </div>
          <Button variant="ghost" size="sm" onClick={() => setAnomaly(null)}>
            Dismiss
          </Button>
        </div>
      )}

      {/* Main Grid Layout */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Left Column - Patient Info & ECG */}
        <div className="xl:col-span-2 space-y-6">
          {/* Patient Info Card */}
          <PatientInfoCard patient={selectedPatient} />

          {/* ECG Waveform */}
          <ECGWaveform 
            data={ecgData} 
            height={250}
            patientId={selectedPatient.id}
          />

          {/* Primary Vitals Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="card-medical p-6 flex flex-col items-center relative">
              <PulseIndicator heartRate={vitals.heartRate} className="mb-4" />
              <GaugeChart 
                value={vitals.heartRate} 
                min={40} 
                max={140} 
                label="Heart Rate" 
                unit="BPM" 
                size="lg"
                variant={vitals.heartRate > 100 ? 'warning' : vitals.heartRate < 60 ? 'danger' : 'success'}
              />
            </div>
            
            <div className="card-medical p-6 flex flex-col items-center">
              <GaugeChart 
                value={vitals.spo2} 
                min={80} 
                max={100} 
                label="SpO₂" 
                unit="%" 
                size="lg" 
                variant={vitals.spo2 < 90 ? 'danger' : vitals.spo2 < 95 ? 'warning' : 'success'}
              />
            </div>
            
            <div className="card-medical p-6 flex flex-col items-center">
              <div className="text-center">
                <Thermometer className={cn(
                  "w-12 h-12 mx-auto mb-4",
                  vitals.temperature > 38.5 ? "text-destructive" : "text-medical-orange"
                )} />
                <p className={cn(
                  "text-4xl font-bold font-mono",
                  vitals.temperature > 38.5 ? "text-destructive" : "text-foreground"
                )}>
                  {vitals.temperature.toFixed(1)}°C
                </p>
                <p className="text-sm text-muted-foreground mt-2">Body Temperature</p>
              </div>
            </div>
          </div>

          {/* Additional Vitals */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="card-medical p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Wind className="w-4 h-4 text-muted-foreground" />
                  <span className="text-sm text-muted-foreground">Respiratory Rate</span>
                </div>
                <span className="text-2xl font-bold font-mono text-foreground">
                  {vitals.respiratoryRate}
                  <span className="text-sm text-muted-foreground ml-1">/min</span>
                </span>
              </div>
            </div>
            <div className="card-medical p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4 text-muted-foreground" />
                  <span className="text-sm text-muted-foreground">Blood Pressure</span>
                </div>
                <span className="text-2xl font-bold font-mono text-foreground">
                  {vitals.systolicBP}/{vitals.diastolicBP}
                  <span className="text-sm text-muted-foreground ml-1">mmHg</span>
                </span>
              </div>
        </div>
      </div>

          {/* Trend Charts - Collapsible */}
          <Collapsible open={showTrends} onOpenChange={setShowTrends}>
            <CollapsibleTrigger asChild>
              <Button variant="outline" className="w-full justify-between">
                <span>Vital Signs Trend Analysis (Last 60 seconds)</span>
                {showTrends ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent className="space-y-4 mt-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <VitalsTrendChart
                  data={vitalsHistory}
                  metric="heartRate"
                  label="Heart Rate"
                  unit="BPM"
                  color="hsl(var(--success))"
                  icon={Heart}
                  min={50}
                  max={150}
                />
                <VitalsTrendChart
                  data={vitalsHistory}
                  metric="spo2"
                  label="SpO₂"
                  unit="%"
                  color="hsl(var(--success))"
                  icon={Wind}
                  min={85}
                  max={100}
                />
                <VitalsTrendChart
                  data={vitalsHistory}
                  metric="temperature"
                  label="Temperature"
                  unit="°C"
                  color="hsl(var(--medical-orange))"
                  icon={Thermometer}
                  min={35}
                  max={40}
                />
                <VitalsTrendChart
                  data={vitalsHistory}
                  metric="respiratoryRate"
                  label="Respiratory Rate"
                  unit="/min"
                  color="hsl(var(--primary))"
                  icon={Activity}
                  min={10}
                  max={30}
                />
              </div>
            </CollapsibleContent>
          </Collapsible>
        </div>

        {/* Right Column - Controls & Devices */}
        <div className="space-y-6">
          {/* ML Risk Prediction Panel */}
          <RiskPredictionPanel
            prediction={prediction}
            riskHistory={riskHistory}
          />

          {/* Anomaly Simulator */}
          <AnomalySimulator
            currentAnomaly={currentAnomaly}
            onTriggerAnomaly={handleTriggerAnomaly}
            onClearAnomaly={handleClearAnomaly}
          />

          {/* Device Status */}
          <div>
            <h3 className="font-display font-semibold text-lg text-foreground mb-4">
              Connected Devices
            </h3>
            <div className="space-y-3">
              {devices.map((device) => (
                <DeviceStatusIndicator key={device.id} device={device} />
              ))}
            </div>
          </div>

          {/* Playback Controls */}
          <div className="card-medical p-4">
            <div className="flex items-center gap-2 mb-3">
              <Settings className="w-4 h-4 text-muted-foreground" />
              <span className="text-sm font-medium text-foreground">Playback</span>
            </div>
            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-xs text-muted-foreground mb-2">
                  <span>Time Range</span>
                  <span>{playbackPosition}s</span>
                </div>
                <Slider
                  value={[playbackPosition]}
                  onValueChange={([value]) => {
                    setPlaybackPosition(value);
                    iomtSimulator.setTimeOffset((60 - value) * 1000);
                  }}
                  min={0}
                  max={60}
                  step={1}
                  className="w-full"
                />
              </div>
              <p className="text-xs text-muted-foreground">
                Adjust slider to review past vitals data
              </p>
        </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default IoMTMonitor;
