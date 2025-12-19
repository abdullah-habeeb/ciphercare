import React, { useState, useMemo, useRef } from 'react';
import { useParams, Link } from 'react-router-dom';
import { HOSPITALS } from '@/lib/constants';
import { 
  ArrowLeft, 
  Heart, 
  Activity, 
  Stethoscope, 
  Users, 
  Building2,
  Upload,
  Play,
  BarChart3,
  Target,
  TrendingUp,
  Database,
  Clock,
  Zap,
  Info,
  MapPin,
  Bed,
  Shield,
  CheckCircle,
  ExternalLink,
  Wifi,
  RefreshCw,
  FileText,
  AlertCircle,
  CheckCircle2,
  Loader2
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { cn } from '@/lib/utils';
import {
  uploadHospitalDataCSV,
  uploadHospitalDataJSON,
  retrainHospitalModel,
  predictHospitalUseCase,
  getModelMetadata,
  getTrainingHistory,
  getMetadataSummary,
  type RetrainResponse,
  type HospitalPredictionResponse,
  type ModelMetadata,
  type MetadataSummary,
  uploadHospitalData,
  retrainHospitalModelNew,
  predictHospitalNew,
  getHospitalMetadata,
  type HospitalUploadResponse,
  type HospitalRetrainResponse,
  type HospitalPredictResponse,
  type HospitalMetadata as HospitalMetadataType
} from '@/lib/api';
import { UseCaseInterface } from '@/components/hospital-usecase/UseCaseInterface';
import { DataInputSection } from '@/components/hospital-usecase/DataInputSection';

const iconMap: Record<string, React.ElementType> = {
  Heart,
  Activity,
  Stethoscope,
  Users,
  Building2,
};

const HospitalDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const hospital = HOSPITALS.find(h => h.id === id);
  const [predictionResult, setPredictionResult] = useState<null | {
    prediction: string;
    confidence: number;
    risk: 'low' | 'medium' | 'high';
  }>(null);
  
  // New state for upload & retrain
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [retraining, setRetraining] = useState(false);
  const [retrainResult, setRetrainResult] = useState<RetrainResponse | HospitalRetrainResponse | null>(null);
  const [showRetrainModal, setShowRetrainModal] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [manualData, setManualData] = useState<Record<string, any>>({});
  
  // New state for use case demo
  const [useCaseInputs, setUseCaseInputs] = useState<Record<string, any>>({});
  const [useCaseResult, setUseCaseResult] = useState<HospitalPredictionResponse | HospitalPredictResponse | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [inferenceHistory, setInferenceHistory] = useState<Array<{timestamp: Date, result: HospitalPredictionResponse | HospitalPredictResponse}>>([]);
  
  // New state for metadata
  const [metadata, setMetadata] = useState<ModelMetadata | HospitalMetadataType | null>(null);
  const [metadataSummary, setMetadataSummary] = useState<MetadataSummary | null>(null);
  const [loadingMetadata, setLoadingMetadata] = useState(false);

  const loadMetadata = React.useCallback(async (hospitalId: string) => {
    setLoadingMetadata(true);
    try {
      // Try new endpoint first
      try {
        const meta = await getHospitalMetadata(hospitalId);
        setMetadata(meta);
      } catch {
        // Fallback to old endpoint
        try {
          const summary = await getMetadataSummary(hospitalId);
          setMetadataSummary(summary);
          const meta = await getModelMetadata(hospitalId);
          setMetadata(meta);
        } catch {
          // Metadata might not exist yet
        }
      }
    } catch (error) {
      console.error('Failed to load metadata:', error);
    } finally {
      setLoadingMetadata(false);
    }
  }, []);

  // Load metadata on mount and when hospital changes
  React.useEffect(() => {
    if (id) {
      loadMetadata(id);
    }
  }, [id, loadMetadata]);

  if (!hospital) {
    return (
      <div className="flex flex-col items-center justify-center h-96">
        <Building2 className="w-16 h-16 text-muted-foreground mb-4" />
        <h2 className="text-xl font-semibold text-foreground">Hospital not found</h2>
        <Link to="/hospitals" className="text-primary hover:underline mt-2">
          Back to Hospitals
        </Link>
      </div>
    );
  }

  const Icon = iconMap[hospital.icon] || Building2;

  const runPrediction = () => {
    // Simulated prediction
    setTimeout(() => {
      setPredictionResult({
        prediction: 'Arrhythmia Detected',
        confidence: 0.87,
        risk: 'medium',
      });
    }, 1500);
  };

  const confusionMatrix = [
    [423, 12],
    [8, 387],
  ];

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-start gap-4">
        <Link to="/hospitals">
          <Button variant="ghost" size="icon">
            <ArrowLeft className="w-5 h-5" />
          </Button>
        </Link>
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center">
              <Icon className="w-6 h-6 text-primary" />
            </div>
            <div>
              <h1 className="text-2xl md:text-3xl font-display font-bold text-foreground">
                {hospital.name}
              </h1>
              <p className="text-muted-foreground">{hospital.fullName}</p>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div className="status-dot status-online status-dot-pulse" />
          <span className="text-sm text-success font-medium">Online</span>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="card-medical p-4">
          <div className="flex items-center gap-2 mb-2">
            <Target className="w-4 h-4 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">Local AUROC</span>
          </div>
          <p className="text-3xl font-bold font-mono text-foreground">{hospital.localAuroc.toFixed(3)}</p>
        </div>
        <div className="card-medical p-4">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp className="w-4 h-4 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">Global AUROC</span>
          </div>
          <p className="text-3xl font-bold font-mono text-primary">{hospital.globalAuroc.toFixed(3)}</p>
        </div>
        <div className="card-medical p-4">
          <div className="flex items-center gap-2 mb-2">
            <Database className="w-4 h-4 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">Samples</span>
          </div>
          <p className="text-3xl font-bold font-mono text-foreground">{hospital.samples.toLocaleString()}</p>
        </div>
        <div className="card-medical p-4">
          <div className="flex items-center gap-2 mb-2">
            <Zap className="w-4 h-4 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">FL Weight</span>
          </div>
          <p className="text-3xl font-bold font-mono text-foreground">{(hospital.contributionWeight * 100).toFixed(0)}%</p>
        </div>
      </div>

      {/* Tabs */}
      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList className="bg-secondary">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="prediction">Prediction Sandbox</TabsTrigger>
          <TabsTrigger value="contribution">FL Contribution</TabsTrigger>
          <TabsTrigger value="iomt">IoMT Integration</TabsTrigger>
          <TabsTrigger value="upload">Upload & Retrain</TabsTrigger>
          <TabsTrigger value="usecase">Use Case Demo</TabsTrigger>
          <TabsTrigger value="metadata">Model Metadata</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="card-medical p-6">
              <h3 className="font-display font-semibold text-lg text-foreground mb-4">Specialty Overview</h3>
              <div className="space-y-4">
                <div>
                  <p className="text-sm text-muted-foreground">Specialty</p>
                  <p className="font-medium text-foreground">{hospital.specialty}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Model Type</p>
                  <p className="font-medium text-foreground">{hospital.modelType}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Description</p>
                  <p className="text-foreground">{hospital.description}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Dataset Characteristics</p>
                  <div className="flex flex-wrap gap-2 mt-2">
                    <span className="px-2 py-1 bg-secondary rounded text-xs">Cleaned</span>
                    <span className="px-2 py-1 bg-secondary rounded text-xs">Balanced</span>
                    <span className="px-2 py-1 bg-secondary rounded text-xs">Annotated</span>
                    <span className="px-2 py-1 bg-secondary rounded text-xs">Quality Verified</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="card-medical p-6">
              <h3 className="font-display font-semibold text-lg text-foreground mb-4">Connection Status</h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Status</span>
                  <span className={cn(
                    "text-sm font-medium flex items-center gap-2",
                    hospital.status === 'online' ? "text-success" : hospital.status === 'warning' ? "text-warning" : "text-destructive"
                  )}>
                    <span className={cn(
                      "status-dot",
                      hospital.status === 'online' ? "status-online" : hospital.status === 'warning' ? "status-warning" : "status-offline"
                    )} />
                    {hospital.status.charAt(0).toUpperCase() + hospital.status.slice(1)}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Location</span>
                  <span className="text-sm text-foreground flex items-center gap-1">
                    <MapPin className="w-3 h-3" />
                    {hospital.location || 'N/A'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Bed Count</span>
                  <span className="text-sm text-foreground flex items-center gap-1">
                    <Bed className="w-3 h-3" />
                    {hospital.bedCount || 'N/A'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Last Updated</span>
                  <span className="text-sm text-foreground">{hospital.lastUpdated}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">FL Rounds</span>
                  <span className="text-sm text-foreground">{hospital.flRoundsCompleted || 8}/10</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Data Quality</span>
                  <span className="text-sm font-medium text-foreground">
                    {hospital.dataQualityScore ? (hospital.dataQualityScore * 100).toFixed(0) + '%' : 'N/A'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Compliance</span>
                  <span className="text-sm text-foreground flex items-center gap-1">
                    <Shield className="w-3 h-3" />
                    {hospital.complianceStatus || 'HIPAA'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Enhanced Metadata Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="card-medical p-4">
              <div className="flex items-center gap-2 mb-2">
                <Activity className="w-4 h-4 text-muted-foreground" />
                <span className="text-sm text-muted-foreground">Connected Devices</span>
              </div>
              <p className="text-2xl font-bold font-mono text-foreground">
                {hospital.connectedDevices || 0}
              </p>
            </div>
            <div className="card-medical p-4">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle className="w-4 h-4 text-muted-foreground" />
                <span className="text-sm text-muted-foreground">Personalized AUROC</span>
              </div>
              <p className="text-2xl font-bold font-mono text-primary">
                {hospital.personalizedAuroc ? hospital.personalizedAuroc.toFixed(3) : (hospital.globalAuroc + hospital.personalizedBoost).toFixed(3)}
              </p>
            </div>
            <div className="card-medical p-4">
              <div className="flex items-center gap-2 mb-2">
                <Clock className="w-4 h-4 text-muted-foreground" />
                <span className="text-sm text-muted-foreground">Last Sync</span>
              </div>
              <p className="text-sm font-medium text-foreground">
                {hospital.lastSyncTime || hospital.lastUpdated}
              </p>
            </div>
          </div>

          {/* Supported Devices */}
          {hospital.supportedDevices && hospital.supportedDevices.length > 0 && (
            <div className="card-medical p-6">
              <h3 className="font-display font-semibold text-lg text-foreground mb-4">Supported IoMT Devices</h3>
              <div className="flex flex-wrap gap-2">
                {hospital.supportedDevices.map((device) => (
                  <span
                    key={device}
                    className="px-3 py-1.5 bg-primary/10 text-primary rounded-lg text-sm font-medium border border-primary/20"
                  >
                    {device}
                  </span>
                ))}
              </div>
            </div>
          )}
        </TabsContent>

        {/* Performance Tab */}
        <TabsContent value="performance" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="card-medical p-6">
              <h3 className="font-display font-semibold text-lg text-foreground mb-4">Model Metrics</h3>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between mb-2">
                    <span className="text-sm text-muted-foreground">Sensitivity</span>
                    <span className="text-sm font-medium text-foreground">{(hospital.sensitivity * 100).toFixed(0)}%</span>
                  </div>
                  <Progress value={hospital.sensitivity * 100} className="h-2" />
                </div>
                <div>
                  <div className="flex justify-between mb-2">
                    <span className="text-sm text-muted-foreground">Specificity</span>
                    <span className="text-sm font-medium text-foreground">{(hospital.specificity * 100).toFixed(0)}%</span>
                  </div>
                  <Progress value={hospital.specificity * 100} className="h-2" />
                </div>
                <div>
                  <div className="flex justify-between mb-2">
                    <span className="text-sm text-muted-foreground">AUROC</span>
                    <span className="text-sm font-medium text-foreground">{(hospital.localAuroc * 100).toFixed(1)}%</span>
                  </div>
                  <Progress value={hospital.localAuroc * 100} className="h-2" />
                </div>
              </div>
            </div>

            <div className="card-medical p-6">
              <h3 className="font-display font-semibold text-lg text-foreground mb-4">Confusion Matrix</h3>
              <div className="grid grid-cols-2 gap-1 max-w-xs">
                <div className="p-4 bg-success/20 rounded-tl-lg text-center">
                  <p className="text-xl font-bold text-success">{confusionMatrix[0][0]}</p>
                  <p className="text-xs text-muted-foreground">True Pos</p>
                </div>
                <div className="p-4 bg-destructive/20 rounded-tr-lg text-center">
                  <p className="text-xl font-bold text-destructive">{confusionMatrix[0][1]}</p>
                  <p className="text-xs text-muted-foreground">False Pos</p>
                </div>
                <div className="p-4 bg-destructive/20 rounded-bl-lg text-center">
                  <p className="text-xl font-bold text-destructive">{confusionMatrix[1][0]}</p>
                  <p className="text-xs text-muted-foreground">False Neg</p>
                </div>
                <div className="p-4 bg-success/20 rounded-br-lg text-center">
                  <p className="text-xl font-bold text-success">{confusionMatrix[1][1]}</p>
                  <p className="text-xs text-muted-foreground">True Neg</p>
                </div>
              </div>
            </div>

            {/* Personalized vs Global */}
            <div className="card-medical p-6 lg:col-span-2">
              <h3 className="font-display font-semibold text-lg text-foreground mb-4">Personalized vs Global Model</h3>
              <div className="grid grid-cols-2 gap-6">
                <div className="p-4 bg-secondary rounded-lg">
                  <p className="text-sm text-muted-foreground mb-2">Global Model</p>
                  <p className="text-3xl font-bold font-mono text-foreground">{hospital.globalAuroc.toFixed(3)}</p>
                  <p className="text-sm text-muted-foreground mt-1">Federated aggregation</p>
                </div>
                <div className="p-4 bg-primary/10 rounded-lg border border-primary/20">
                  <p className="text-sm text-muted-foreground mb-2">Personalized Model</p>
                  <p className="text-3xl font-bold font-mono text-primary">
                    {(hospital.globalAuroc + hospital.personalizedBoost).toFixed(3)}
                  </p>
                  <p className="text-sm text-success mt-1">+{(hospital.personalizedBoost * 100).toFixed(1)}% from fine-tuning</p>
                </div>
              </div>
            </div>
          </div>
        </TabsContent>

        {/* Prediction Sandbox Tab */}
        <TabsContent value="prediction" className="space-y-6">
          <div className="card-medical p-6">
            <h3 className="font-display font-semibold text-lg text-foreground mb-4">
              Prediction Sandbox
              <span className="ml-2 px-2 py-1 bg-primary/10 text-primary text-xs rounded">Demo Mode</span>
            </h3>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Input Section */}
              <div className="space-y-4">
                <div className="border-2 border-dashed border-border rounded-lg p-8 text-center hover:border-primary/50 transition-colors">
                  <Upload className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                  <p className="font-medium text-foreground mb-2">
                    {hospital.id === 'A' || hospital.id === 'D' ? 'Upload ECG Data' : 
                     hospital.id === 'C' ? 'Upload Chest X-Ray' :
                     hospital.id === 'B' ? 'Enter Vital Signs' :
                     'Upload Multi-Modal Bundle'}
                  </p>
                  <p className="text-sm text-muted-foreground mb-4">
                    Drag and drop or click to upload
                  </p>
                  <Button variant="outline">Select File</Button>
                </div>

                {hospital.id === 'B' && (
                  <div className="space-y-3">
                    <p className="text-sm font-medium text-foreground">Or enter vitals manually:</p>
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="text-xs text-muted-foreground">Heart Rate (BPM)</label>
                        <input type="number" placeholder="72" className="w-full mt-1 px-3 py-2 border border-border rounded-lg bg-background" />
                      </div>
                      <div>
                        <label className="text-xs text-muted-foreground">SpO₂ (%)</label>
                        <input type="number" placeholder="98" className="w-full mt-1 px-3 py-2 border border-border rounded-lg bg-background" />
                      </div>
                      <div>
                        <label className="text-xs text-muted-foreground">Temperature (°C)</label>
                        <input type="number" placeholder="36.6" className="w-full mt-1 px-3 py-2 border border-border rounded-lg bg-background" />
                      </div>
                      <div>
                        <label className="text-xs text-muted-foreground">Blood Pressure</label>
                        <input type="text" placeholder="120/80" className="w-full mt-1 px-3 py-2 border border-border rounded-lg bg-background" />
                      </div>
                    </div>
                  </div>
                )}

                <Button className="w-full" onClick={runPrediction}>
                  <Play className="w-4 h-4 mr-2" />
                  Run Prediction
                </Button>
              </div>

              {/* Results Section */}
              <div className="space-y-4">
                <h4 className="font-medium text-foreground">Prediction Results</h4>
                {predictionResult ? (
                  <div className="space-y-4">
                    <div className={cn(
                      "p-4 rounded-lg border",
                      predictionResult.risk === 'low' && "bg-success/10 border-success/20",
                      predictionResult.risk === 'medium' && "bg-warning/10 border-warning/20",
                      predictionResult.risk === 'high' && "bg-destructive/10 border-destructive/20"
                    )}>
                      <p className="text-sm text-muted-foreground">Prediction</p>
                      <p className="text-xl font-bold text-foreground">{predictionResult.prediction}</p>
                    </div>
                    <div className="p-4 bg-secondary rounded-lg">
                      <p className="text-sm text-muted-foreground">Confidence Score</p>
                      <p className="text-3xl font-bold font-mono text-primary">{(predictionResult.confidence * 100).toFixed(1)}%</p>
                      <Progress value={predictionResult.confidence * 100} className="h-2 mt-2" />
                    </div>
                    <div className="p-4 bg-secondary rounded-lg">
                      <p className="text-sm text-muted-foreground mb-2">Feature Importance</p>
                      <div className="space-y-2">
                        {['QRS Duration', 'Heart Rate Variability', 'ST Elevation', 'P-Wave'].map((feature, i) => (
                          <div key={feature} className="flex items-center gap-2">
                            <span className="text-xs text-muted-foreground w-32">{feature}</span>
                            <Progress value={90 - i * 15} className="h-1.5 flex-1" />
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="p-8 bg-secondary rounded-lg text-center">
                    <BarChart3 className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                    <p className="text-muted-foreground">Run a prediction to see results</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </TabsContent>

        {/* FL Contribution Tab */}
        <TabsContent value="contribution" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="card-medical p-6">
              <h3 className="font-display font-semibold text-lg text-foreground mb-4">FL Contribution Metrics</h3>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between mb-2">
                    <span className="text-sm text-muted-foreground">Contribution Weight</span>
                    <span className="text-sm font-medium text-foreground">{(hospital.contributionWeight * 100).toFixed(0)}%</span>
                  </div>
                  <Progress value={hospital.contributionWeight * 100} className="h-2" />
                  <p className="text-xs text-muted-foreground mt-1">
                    w = 0.6 × AUROC² + 0.4 × (n / Σn)
                  </p>
                </div>
                <div>
                  <div className="flex justify-between mb-2">
                    <span className="text-sm text-muted-foreground">Update Magnitude</span>
                    <span className="text-sm font-medium text-foreground">0.0234</span>
                  </div>
                  <Progress value={23.4} className="h-2" />
                </div>
                <div>
                  <div className="flex justify-between mb-2">
                    <span className="text-sm text-muted-foreground">Drift from Global</span>
                    <span className={cn(
                      "text-sm font-medium",
                      hospital.driftMagnitude > 0.03 ? "text-warning" : "text-success"
                    )}>
                      {hospital.driftMagnitude.toFixed(3)}
                    </span>
                  </div>
                  <Progress value={hospital.driftMagnitude * 1000} className="h-2" />
                </div>
              </div>
            </div>

            <div className="card-medical p-6">
              <h3 className="font-display font-semibold text-lg text-foreground mb-4">Personalization Status</h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-3 bg-success/10 rounded-lg border border-success/20">
                  <span className="text-sm font-medium text-foreground">Fine-tuning Active</span>
                  <span className="status-dot status-online" />
                </div>
                <div className="p-4 bg-secondary rounded-lg">
                  <p className="text-sm text-muted-foreground">Personalization Boost</p>
                  <p className="text-2xl font-bold font-mono text-success">+{(hospital.personalizedBoost * 100).toFixed(1)}%</p>
                  <p className="text-xs text-muted-foreground mt-1">AUROC improvement from local fine-tuning</p>
                </div>
                <div className="p-4 bg-secondary rounded-lg">
                  <p className="text-sm text-muted-foreground">FedProx μ Parameter</p>
                  <p className="text-2xl font-bold font-mono text-foreground">0.01</p>
                  <p className="text-xs text-muted-foreground mt-1">Proximal regularization strength</p>
                </div>
              </div>
            </div>
          </div>
        </TabsContent>

        {/* IoMT Integration Tab */}
        <TabsContent value="iomt" className="space-y-6">
          <div className="card-medical p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="font-display font-semibold text-lg text-foreground">IoMT Device Integration</h3>
                <p className="text-sm text-muted-foreground mt-1">
                  Real-time monitoring and device management for {hospital.name}
                </p>
              </div>
              <Button asChild variant="outline" className="gap-2">
                <Link to="/iomt-monitor">
                  <ExternalLink className="w-4 h-4" />
                  View Live Monitor
                </Link>
              </Button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
              <div className="p-4 bg-secondary rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <Wifi className="w-4 h-4 text-success" />
                  <span className="text-sm text-muted-foreground">Active Devices</span>
                </div>
                <p className="text-2xl font-bold font-mono text-foreground">
                  {hospital.connectedDevices || 0}
                </p>
                <p className="text-xs text-muted-foreground mt-1">Currently streaming</p>
              </div>
              <div className="p-4 bg-secondary rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <Activity className="w-4 h-4 text-primary" />
                  <span className="text-sm text-muted-foreground">Device Types</span>
                </div>
                <p className="text-2xl font-bold font-mono text-foreground">
                  {hospital.supportedDevices?.length || 0}
                </p>
                <p className="text-xs text-muted-foreground mt-1">Supported types</p>
              </div>
              <div className="p-4 bg-secondary rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <Clock className="w-4 h-4 text-muted-foreground" />
                  <span className="text-sm text-muted-foreground">Last Update</span>
                </div>
                <p className="text-sm font-medium text-foreground">
                  {hospital.lastSyncTime || hospital.lastUpdated}
                </p>
                <p className="text-xs text-muted-foreground mt-1">Data sync status</p>
              </div>
            </div>

            {hospital.supportedDevices && hospital.supportedDevices.length > 0 && (
              <div>
                <h4 className="font-medium text-foreground mb-3">Device Capabilities</h4>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  {hospital.supportedDevices.map((device) => {
                    const deviceIcons: Record<string, React.ElementType> = {
                      ECG: Heart,
                      PulseOx: Activity,
                      BP: Activity,
                      Temp: Thermometer,
                      Respiratory: Wind,
                      XRay: Stethoscope,
                    };
                    const DeviceIcon = deviceIcons[device] || Activity;
                    
                    return (
                      <div
                        key={device}
                        className="p-3 bg-primary/5 border border-primary/20 rounded-lg flex items-center gap-2"
                      >
                        <DeviceIcon className="w-4 h-4 text-primary" />
                        <span className="text-sm font-medium text-foreground">{device}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            <div className="mt-6 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <div className="flex items-start gap-3">
                <Info className="w-5 h-5 text-primary mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-foreground mb-1">
                    IoMT Data Integration
                  </p>
                  <p className="text-sm text-muted-foreground">
                    All device data from {hospital.name} is securely streamed to the CipherCare platform 
                    for real-time analysis. Data is processed locally before federated aggregation, ensuring 
                    complete privacy compliance ({hospital.complianceStatus || 'HIPAA'}).
                  </p>
                </div>
              </div>
            </div>
          </div>
        </TabsContent>

        {/* Upload Data & Retrain Tab */}
        <TabsContent value="upload" className="space-y-6">
          <div className="card-medical p-6">
            <h3 className="font-display font-semibold text-lg text-foreground mb-4">
              Upload Data & Retrain Model
            </h3>
            
            <div className="space-y-6">
              {/* CSV Upload Section */}
              <div>
                <Label className="text-sm font-medium text-foreground mb-2 block">
                  Upload CSV Data
                </Label>
                <div
                  className={cn(
                    "border-2 border-dashed rounded-lg p-8 text-center transition-colors",
                    uploadFile ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"
                  )}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".csv"
                    className="hidden"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) setUploadFile(file);
                    }}
                  />
                  <Upload className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                  {uploadFile ? (
                    <div>
                      <p className="font-medium text-foreground">{uploadFile.name}</p>
                      <p className="text-sm text-muted-foreground mt-1">
                        {(uploadFile.size / 1024).toFixed(2)} KB
                      </p>
                    </div>
                  ) : (
                    <>
                      <p className="font-medium text-foreground mb-2">
                        Drag and drop CSV file or click to browse
                      </p>
                      <p className="text-sm text-muted-foreground">
                        CSV should contain patient vitals and features
                      </p>
                    </>
                  )}
                </div>
                <Button
                  className="w-full mt-4"
                  onClick={async () => {
                    if (!uploadFile || !id) return;
                    setUploading(true);
                    try {
                      await uploadHospitalData(id, uploadFile);
                      setUploadFile(null);
                      if (fileInputRef.current) fileInputRef.current.value = '';
                      alert('Data uploaded successfully!');
                    } catch (error: any) {
                      alert(`Upload failed: ${error.message}`);
                    } finally {
                      setUploading(false);
                    }
                  }}
                  disabled={!uploadFile || uploading}
                >
                  {uploading ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Uploading...
                    </>
                  ) : (
                    <>
                      <Upload className="w-4 h-4 mr-2" />
                      Send Data to Database
                    </>
                  )}
                </Button>
              </div>

              {/* Manual Data Entry */}
              <div>
                <Label className="text-sm font-medium text-foreground mb-2 block">
                  Manual Data Entry
                </Label>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <Label className="text-xs text-muted-foreground">Heart Rate</Label>
                    <Input
                      type="number"
                      placeholder="72"
                      onChange={(e) => setUseCaseInputs({...useCaseInputs, heartRate: parseFloat(e.target.value)})}
                    />
                  </div>
                  <div>
                    <Label className="text-xs text-muted-foreground">SpO₂ (%)</Label>
                    <Input
                      type="number"
                      placeholder="98"
                      onChange={(e) => setUseCaseInputs({...useCaseInputs, spo2: parseFloat(e.target.value)})}
                    />
                  </div>
                  <div>
                    <Label className="text-xs text-muted-foreground">Temperature (°C)</Label>
                    <Input
                      type="number"
                      placeholder="36.6"
                      onChange={(e) => setUseCaseInputs({...useCaseInputs, temperature: parseFloat(e.target.value)})}
                    />
                  </div>
                  <div>
                    <Label className="text-xs text-muted-foreground">Respiratory Rate</Label>
                    <Input
                      type="number"
                      placeholder="16"
                      onChange={(e) => setUseCaseInputs({...useCaseInputs, respRate: parseFloat(e.target.value)})}
                    />
                  </div>
                </div>
                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <Label className="text-xs text-muted-foreground">Heart Rate</Label>
                      <Input
                        type="number"
                        placeholder="72"
                        value={manualData.heartRate || ''}
                        onChange={(e) => setManualData({...manualData, heartRate: parseFloat(e.target.value) || 0})}
                      />
                    </div>
                    <div>
                      <Label className="text-xs text-muted-foreground">SpO₂ (%)</Label>
                      <Input
                        type="number"
                        placeholder="98"
                        value={manualData.spo2 || ''}
                        onChange={(e) => setManualData({...manualData, spo2: parseFloat(e.target.value) || 0})}
                      />
                    </div>
                    <div>
                      <Label className="text-xs text-muted-foreground">Temperature (°C)</Label>
                      <Input
                        type="number"
                        placeholder="36.6"
                        value={manualData.temperature || ''}
                        onChange={(e) => setManualData({...manualData, temperature: parseFloat(e.target.value) || 0})}
                      />
                    </div>
                    <div>
                      <Label className="text-xs text-muted-foreground">Respiratory Rate</Label>
                      <Input
                        type="number"
                        placeholder="16"
                        value={manualData.respRate || ''}
                        onChange={(e) => setManualData({...manualData, respRate: parseFloat(e.target.value) || 0})}
                      />
                    </div>
                  </div>
                  <Button
                    className="w-full"
                    variant="outline"
                    onClick={async () => {
                      if (!id) return;
                      setUploading(true);
                      try {
                        await uploadHospitalData(id, undefined, [manualData]);
                        setManualData({});
                        alert('Data uploaded successfully!');
                      } catch (error: any) {
                        alert(`Upload failed: ${error.message}`);
                      } finally {
                        setUploading(false);
                      }
                    }}
                    disabled={Object.keys(manualData).length === 0 || uploading}
                  >
                    <Database className="w-4 h-4 mr-2" />
                    Save to Database
                  </Button>
                </div>
              </div>

              {/* Retrain Button */}
              <div className="border-t pt-6">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h4 className="font-medium text-foreground">Model Retraining</h4>
                    <p className="text-sm text-muted-foreground">
                      Retrain model with latest hospital data
                    </p>
                  </div>
                  <Button
                    onClick={async () => {
                      if (!id) return;
                      setRetraining(true);
                      setShowRetrainModal(true);
                      try {
                        // Try new endpoint first
                        let result;
                        try {
                          result = await retrainHospitalModelNew(id);
                        } catch {
                          // Fallback to old endpoint
                          result = await retrainHospitalModel(id);
                        }
                        setRetrainResult(result);
                        // Refresh metadata after training
                        await loadMetadata(id);
                      } catch (error: any) {
                        alert(`Retraining failed: ${error.message}`);
                      } finally {
                        setRetraining(false);
                      }
                    }}
                    disabled={retraining}
                  >
                    {retraining ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Training...
                      </>
                    ) : (
                      <>
                        <RefreshCw className="w-4 h-4 mr-2" />
                        Retrain Model
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </div>
          </div>

          {/* Retrain Modal */}
          <Dialog open={showRetrainModal} onOpenChange={setShowRetrainModal}>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Model Retraining</DialogTitle>
                <DialogDescription>
                  {retraining ? 'Training model with latest data...' : retrainResult ? 'Training completed!' : 'Ready to retrain'}
                </DialogDescription>
              </DialogHeader>
              {retraining ? (
                <div className="space-y-4">
                  <Progress value={75} className="h-2" />
                  <p className="text-sm text-muted-foreground text-center">
                    Processing data and training model...
                  </p>
                </div>
              ) : retrainResult ? (
                <div className="space-y-4">
                  <div className="p-4 bg-success/10 border border-success/20 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <CheckCircle2 className="w-5 h-5 text-success" />
                      <span className="font-medium text-foreground">Training Complete</span>
                    </div>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Local AUROC:</span>
                        <span className="font-mono font-bold text-foreground">
                          {retrainResult.local_auroc.toFixed(4)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Samples Used:</span>
                        <span className="font-mono text-foreground">
                          {retrainResult.samples_used.toLocaleString()}
                        </span>
                      </div>
                      {retrainResult.improvement !== null && (
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Improvement:</span>
                          <span className={cn(
                            "font-mono font-bold",
                            retrainResult.improvement > 0 ? "text-success" : "text-destructive"
                          )}>
                            {retrainResult.improvement > 0 ? '+' : ''}{retrainResult.improvement.toFixed(4)}
                          </span>
                        </div>
                      )}
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Duration:</span>
                        <span className="font-mono text-foreground">
                          {retrainResult.training_duration_seconds.toFixed(1)}s
                        </span>
                      </div>
                    </div>
                  </div>
                  <Button
                    className="w-full"
                    onClick={() => {
                      setShowRetrainModal(false);
                      setRetrainResult(null);
                      // Refresh metadata
                      if (id) {
                        loadMetadata(id);
                      }
                    }}
                  >
                    Close
                  </Button>
                </div>
              ) : null}
            </DialogContent>
          </Dialog>
        </TabsContent>

        {/* Hospital Use Case Demonstration Tab */}
        <TabsContent value="usecase" className="space-y-6">
          <div className="card-medical p-6">
            <h3 className="font-display font-semibold text-lg text-foreground mb-4">
              Hospital Use Case Demonstration
            </h3>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Input Form */}
              <div className="space-y-4">
                <h4 className="font-medium text-foreground">Input Data</h4>
                {hospital.id === 'A' || hospital.id === 'D' ? (
                  <div className="space-y-3">
                    <div>
                      <Label>ECG QRS Duration (ms)</Label>
                      <Input
                        type="number"
                        placeholder="80"
                        onChange={(e) => setUseCaseInputs({...useCaseInputs, qrs_duration: parseFloat(e.target.value)})}
                      />
                    </div>
                    <div>
                      <Label>Heart Rate Variability</Label>
                      <Input
                        type="number"
                        placeholder="45"
                        onChange={(e) => setUseCaseInputs({...useCaseInputs, hrv: parseFloat(e.target.value)})}
                      />
                    </div>
                    <div>
                      <Label>ST Elevation (mm)</Label>
                      <Input
                        type="number"
                        placeholder="0.5"
                        onChange={(e) => setUseCaseInputs({...useCaseInputs, st_elevation: parseFloat(e.target.value)})}
                      />
                    </div>
                    <div>
                      <Label>Heart Rate (BPM)</Label>
                      <Input
                        type="number"
                        placeholder="72"
                        onChange={(e) => setUseCaseInputs({...useCaseInputs, heartRate: parseFloat(e.target.value)})}
                      />
                    </div>
                  </div>
                ) : hospital.id === 'B' ? (
                  <div className="space-y-3">
                    <div>
                      <Label>Heart Rate (BPM)</Label>
                      <Input
                        type="number"
                        placeholder="72"
                        onChange={(e) => setUseCaseInputs({...useCaseInputs, heartRate: parseFloat(e.target.value)})}
                      />
                    </div>
                    <div>
                      <Label>SpO₂ (%)</Label>
                      <Input
                        type="number"
                        placeholder="98"
                        onChange={(e) => setUseCaseInputs({...useCaseInputs, spo2: parseFloat(e.target.value)})}
                      />
                    </div>
                    <div>
                      <Label>Temperature (°C)</Label>
                      <Input
                        type="number"
                        placeholder="36.6"
                        onChange={(e) => setUseCaseInputs({...useCaseInputs, temperature: parseFloat(e.target.value)})}
                      />
                    </div>
                    <div>
                      <Label>Respiratory Rate</Label>
                      <Input
                        type="number"
                        placeholder="16"
                        onChange={(e) => setUseCaseInputs({...useCaseInputs, respRate: parseFloat(e.target.value)})}
                      />
                    </div>
                    <div>
                      <Label>Systolic BP</Label>
                      <Input
                        type="number"
                        placeholder="120"
                        onChange={(e) => setUseCaseInputs({...useCaseInputs, systolicBP: parseFloat(e.target.value)})}
                      />
                    </div>
                    <div>
                      <Label>Diastolic BP</Label>
                      <Input
                        type="number"
                        placeholder="80"
                        onChange={(e) => setUseCaseInputs({...useCaseInputs, diastolicBP: parseFloat(e.target.value)})}
                      />
                    </div>
                  </div>
                ) : hospital.id === 'C' ? (
                  <div className="space-y-3">
                    <div>
                      <Label>X-Ray Opacity Score</Label>
                      <Input
                        type="number"
                        placeholder="0.5"
                        onChange={(e) => setUseCaseInputs({...useCaseInputs, opacity_score: parseFloat(e.target.value)})}
                      />
                    </div>
                    <div>
                      <Label>Lung Area Affected (%)</Label>
                      <Input
                        type="number"
                        placeholder="15"
                        onChange={(e) => setUseCaseInputs({...useCaseInputs, lung_area: parseFloat(e.target.value)})}
                      />
                    </div>
                    <div>
                      <Label>Respiratory Rate</Label>
                      <Input
                        type="number"
                        placeholder="18"
                        onChange={(e) => setUseCaseInputs({...useCaseInputs, respRate: parseFloat(e.target.value)})}
                      />
                    </div>
                    <div>
                      <Label>SpO₂ (%)</Label>
                      <Input
                        type="number"
                        placeholder="96"
                        onChange={(e) => setUseCaseInputs({...useCaseInputs, spo2: parseFloat(e.target.value)})}
                      />
                    </div>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <div>
                      <Label>Heart Rate (BPM)</Label>
                      <Input
                        type="number"
                        placeholder="72"
                        onChange={(e) => setUseCaseInputs({...useCaseInputs, heartRate: parseFloat(e.target.value)})}
                      />
                    </div>
                    <div>
                      <Label>SpO₂ (%)</Label>
                      <Input
                        type="number"
                        placeholder="98"
                        onChange={(e) => setUseCaseInputs({...useCaseInputs, spo2: parseFloat(e.target.value)})}
                      />
                    </div>
                    <div>
                      <Label>Temperature (°C)</Label>
                      <Input
                        type="number"
                        placeholder="36.6"
                        onChange={(e) => setUseCaseInputs({...useCaseInputs, temperature: parseFloat(e.target.value)})}
                      />
                    </div>
                    <div>
                      <Label>X-Ray Score</Label>
                      <Input
                        type="number"
                        placeholder="0.3"
                        onChange={(e) => setUseCaseInputs({...useCaseInputs, xray_score: parseFloat(e.target.value)})}
                      />
                    </div>
                  </div>
                )}
                <Button
                  className="w-full"
                  onClick={async () => {
                    if (!id) return;
                    setPredicting(true);
                    try {
                      // Try new endpoint first
                      let result;
                      try {
                        result = await predictHospitalNew(id, useCaseInputs);
                      } catch {
                        // Fallback to old endpoint
                        result = await predictHospitalUseCase(id, useCaseInputs);
                      }
                      setUseCaseResult(result);
                      setInferenceHistory(prev => [
                        {timestamp: new Date(), result},
                        ...prev.slice(0, 4)
                      ]);
                    } catch (error: any) {
                      alert(`Prediction failed: ${error.message}`);
                    } finally {
                      setPredicting(false);
                    }
                  }}
                  disabled={Object.keys(useCaseInputs).length === 0 || predicting}
                >
                  {predicting ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Running Prediction...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 mr-2" />
                      Run Prediction
                    </>
                  )}
                </Button>
              </div>

              {/* Results */}
              <div className="space-y-4">
                <h4 className="font-medium text-foreground">Prediction Results</h4>
                {useCaseResult ? (
                  <div className="space-y-4">
                    <div className={cn(
                      "p-4 rounded-lg border",
                      useCaseResult.severity === 'stable' && "bg-success/10 border-success/20",
                      useCaseResult.severity === 'moderate' && "bg-warning/10 border-warning/20",
                      useCaseResult.severity === 'critical' && "bg-destructive/10 border-destructive/20"
                    )}>
                      <p className="text-sm text-muted-foreground mb-1">Severity</p>
                      <p className="text-xl font-bold text-foreground capitalize">
                        {useCaseResult.severity}
                      </p>
                    </div>
                    <div className="p-4 bg-secondary rounded-lg">
                      <p className="text-sm text-muted-foreground mb-1">Risk Score</p>
                      <p className="text-3xl font-bold font-mono text-primary">
                        {(useCaseResult.risk_score * 100).toFixed(1)}%
                      </p>
                      <Progress value={useCaseResult.risk_score * 100} className="h-2 mt-2" />
                    </div>
                    <div className="p-4 bg-secondary rounded-lg">
                      <p className="text-sm text-muted-foreground mb-2">Recommended Action</p>
                      <p className="text-foreground">{useCaseResult.recommended_action}</p>
                    </div>
                    <div className="p-4 bg-secondary rounded-lg">
                      <p className="text-sm text-muted-foreground mb-2">Feature Contributions</p>
                      <div className="space-y-2">
                        {Object.entries(useCaseResult.explanation).map(([key, value]) => (
                          <div key={key} className="flex items-center gap-2">
                            <span className="text-xs text-muted-foreground w-32 capitalize">
                              {key.replace(/([A-Z])/g, ' $1').trim()}
                            </span>
                            <Progress value={Math.abs(parseFloat(value.replace('%', '')) || 0)} className="h-1.5 flex-1" />
                            <span className="text-xs font-mono text-foreground w-16 text-right">{value}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="p-8 bg-secondary rounded-lg text-center">
                    <BarChart3 className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                    <p className="text-muted-foreground">Run a prediction to see results</p>
                  </div>
                )}

                {/* Inference History Chart */}
                {inferenceHistory.length > 0 && (
                  <div className="p-4 bg-secondary rounded-lg">
                    <p className="text-sm text-muted-foreground mb-3">Last 5 Inferences</p>
                    <div className="space-y-2">
                      {inferenceHistory.map((item, idx) => (
                        <div key={idx} className="flex items-center gap-2">
                          <span className="text-xs text-muted-foreground w-20">
                            {item.timestamp.toLocaleTimeString()}
                          </span>
                          <Progress
                            value={item.result.risk_score * 100}
                            className="h-2 flex-1"
                          />
                          <span className={cn(
                            "text-xs font-mono w-16 text-right",
                            item.result.severity === 'critical' && "text-destructive",
                            item.result.severity === 'moderate' && "text-warning",
                            item.result.severity === 'stable' && "text-success"
                          )}>
                            {(item.result.risk_score * 100).toFixed(0)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </TabsContent>

        {/* Model Metadata Tab */}
        <TabsContent value="metadata" className="space-y-6">
          <div className="card-medical p-6">
            <h3 className="font-display font-semibold text-lg text-foreground mb-4">
              Model Metadata & Performance Tracking
            </h3>
            
            {loadingMetadata ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
              </div>
            ) : metadata ? (
              <div className="space-y-6">
                {/* Current Metrics Grid */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="p-4 bg-secondary rounded-lg">
                    <p className="text-xs text-muted-foreground mb-1">Local AUROC</p>
                    <p className="text-2xl font-bold font-mono text-primary">
                      {('local_auroc' in metadata ? metadata.local_auroc : (metadata as any).localAuroc)?.toFixed(3) || 'N/A'}
                    </p>
                  </div>
                  <div className="p-4 bg-secondary rounded-lg">
                    <p className="text-xs text-muted-foreground mb-1">Training Samples</p>
                    <p className="text-2xl font-bold font-mono text-foreground">
                      {('samples' in metadata ? metadata.samples : (metadata as any).training_samples)?.toLocaleString() || 'N/A'}
                    </p>
                  </div>
                  <div className="p-4 bg-secondary rounded-lg">
                    <p className="text-xs text-muted-foreground mb-1">Drift Score</p>
                    <p className={cn(
                      "text-2xl font-bold font-mono",
                      (('drift_score' in metadata ? metadata.drift_score : (metadata as any).drift_score) || 0) > 0.05
                        ? "text-warning"
                        : "text-success"
                    )}>
                      {('drift_score' in metadata ? metadata.drift_score : (metadata as any).drift_score)?.toFixed(3) || '0.000'}
                    </p>
                  </div>
                  <div className="p-4 bg-secondary rounded-lg">
                    <p className="text-xs text-muted-foreground mb-1">Model Path</p>
                    <p className="text-xs font-medium text-foreground truncate">
                      {('model_path' in metadata ? metadata.model_path : 'N/A')}
                    </p>
                  </div>
                </div>

                {/* Last Trained */}
                {('last_trained_at' in metadata && metadata.last_trained_at) && (
                  <div className="p-4 bg-primary/5 border border-primary/20 rounded-lg">
                    <div className="flex items-center gap-2 mb-1">
                      <Clock className="w-4 h-4 text-primary" />
                      <span className="text-sm font-medium text-foreground">Last Trained</span>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      {new Date(metadata.last_trained_at).toLocaleString()}
                    </p>
                  </div>
                )}

                {/* Status Badges */}
                <div className="flex gap-2">
                  <span className={cn(
                    "px-3 py-1 rounded-full text-xs font-medium",
                    (('local_auroc' in metadata ? metadata.local_auroc : (metadata as any).localAuroc) || 0) > 0.9
                      ? "bg-success/20 text-success"
                      : (('local_auroc' in metadata ? metadata.local_auroc : (metadata as any).localAuroc) || 0) > 0.8
                      ? "bg-warning/20 text-warning"
                      : "bg-destructive/20 text-destructive"
                  )}>
                    {((('local_auroc' in metadata ? metadata.local_auroc : (metadata as any).localAuroc) || 0) > 0.9) ? 'Excellent' :
                     ((('local_auroc' in metadata ? metadata.local_auroc : (metadata as any).localAuroc) || 0) > 0.8) ? 'Good' : 'Needs Improvement'}
                  </span>
                  <span className={cn(
                    "px-3 py-1 rounded-full text-xs font-medium",
                    (('drift_score' in metadata ? metadata.drift_score : (metadata as any).drift_score) || 0) < 0.05
                      ? "bg-success/20 text-success"
                      : (('drift_score' in metadata ? metadata.drift_score : (metadata as any).drift_score) || 0) < 0.1
                      ? "bg-warning/20 text-warning"
                      : "bg-destructive/20 text-destructive"
                  )}>
                    Drift: {((('drift_score' in metadata ? metadata.drift_score : (metadata as any).drift_score) || 0) < 0.05) ? 'Low' :
                           ((('drift_score' in metadata ? metadata.drift_score : (metadata as any).drift_score) || 0) < 0.1) ? 'Moderate' : 'High'}
                  </span>
                </div>

                <Button
                  variant="outline"
                  onClick={() => {
                    if (id) loadMetadata(id);
                  }}
                >
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Refresh Metadata
                </Button>
              </div>
            ) : (
              <div className="text-center py-12">
                <AlertCircle className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-muted-foreground">No metadata available. Train a model first.</p>
              </div>
            )}
          </div>
        </TabsContent>
      </Tabs>

      {/* Hospital Use-Case Interface Section */}
      <div className="mt-8">
        <UseCaseInterface 
          hospitalId={hospital.id} 
          hospitalName={hospital.name}
        />
      </div>

      {/* Test Data Input & Prediction Section */}
      <div className="mt-8">
        <DataInputSection 
          hospitalId={hospital.id} 
          hospitalName={hospital.name}
        />
      </div>
    </div>
  );
};

export default HospitalDetail;
