import { useState, useEffect } from 'react';
import { Heart, Activity, Wind, Thermometer, Footprints, Wifi, Siren, Play, Pause, Keyboard, Zap, CheckCircle, AlertTriangle } from 'lucide-react';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

// Simulated streaming data generator
const generateVital = (base, variance) => base + (Math.random() - 0.5) * variance;

export default function EmergencyDemo() {
    const [mode, setMode] = useState('STREAM'); // STREAM | MANUAL
    const [isStreaming, setIsStreaming] = useState(true);

    // Streaming State
    const [vitals, setVitals] = useState({
        hr: 72, hrHistory: Array(40).fill(72),
        spo2: 97.2, spo2History: Array(40).fill(97.2),
        temp: 36.6, tempHistory: Array(40).fill(36.6),
        steps: 12, stepsHistory: Array(40).fill(12),
    });
    const [streamPrediction, setStreamPrediction] = useState(null);

    // Manual State
    const [manualInputs, setManualInputs] = useState({
        heart_rate: 72,
        blood_pressure_sys: 120,
        blood_pressure_dia: 80,
        oxygen_saturation: 98,
        features: []
    });
    const [manualResult, setManualResult] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);

    // Stream Effect
    useEffect(() => {
        if (mode !== 'STREAM' || !isStreaming) return;

        const interval = setInterval(() => {
            setVitals(prev => {
                const newHr = Math.round(generateVital(72, 20));
                const newSpo2 = parseFloat(generateVital(97, 3).toFixed(1));
                const newTemp = parseFloat(generateVital(36.6, 0.8).toFixed(1));
                const newSteps = Math.round(generateVital(12, 8));

                // Simple risk heuristic for stream
                let risk = 0.1;
                if (newHr > 100 || newHr < 55) risk += 0.35;
                if (newSpo2 < 94) risk += 0.4;
                const level = risk > 0.6 ? 'Critical' : risk > 0.35 ? 'Warning' : 'Stable';
                const color = risk > 0.6 ? 'red' : risk > 0.35 ? 'yellow' : 'green';
                setStreamPrediction({ risk: Math.min(risk, 0.95), level, color });

                return {
                    hr: newHr,
                    hrHistory: [...prev.hrHistory.slice(1), newHr],
                    spo2: newSpo2,
                    spo2History: [...prev.spo2History.slice(1), newSpo2],
                    temp: newTemp,
                    tempHistory: [...prev.tempHistory.slice(1), newTemp],
                    steps: Math.max(0, newSteps),
                    stepsHistory: [...prev.stepsHistory.slice(1), Math.max(0, newSteps)],
                };
            });
        }, 800);

        return () => clearInterval(interval);
    }, [mode, isStreaming]);

    const handleManualAnalyze = async () => {
        setIsAnalyzing(true);
        try {
            const res = await axios.post(`${API_URL}/api/inference/hospital_b`, manualInputs);
            setManualResult(res.data);
        } catch (err) {
            console.error(err);
        }
        setIsAnalyzing(false);
    };

    return (
        <div className="p-8 space-y-6 h-full overflow-hidden flex flex-col">
            {/* Header */}
            <div className="flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold text-white flex items-center gap-3">
                        <Siren className="text-red-500" /> IoMT Feature Streams
                    </h1>
                    <p className="text-gray-400 mt-1">Real-time Patient Monitoring & Risk Inference</p>
                </div>

                {/* Toggle */}
                <div className="flex bg-dark-card p-1 rounded-lg border border-white/10">
                    <button
                        onClick={() => setMode('STREAM')}
                        className={`flex items-center gap-2 px-4 py-2 rounded-md font-medium text-sm transition-all ${mode === 'STREAM' ? 'bg-blue-600 text-white shadow-lg' : 'text-gray-400 hover:text-white'
                            }`}
                    >
                        <Wifi size={16} /> Live Stream
                    </button>
                    <button
                        onClick={() => setMode('MANUAL')}
                        className={`flex items-center gap-2 px-4 py-2 rounded-md font-medium text-sm transition-all ${mode === 'MANUAL' ? 'bg-purple-600 text-white shadow-lg' : 'text-gray-400 hover:text-white'
                            }`}
                    >
                        <Keyboard size={16} /> Manual Entry
                    </button>
                </div>
            </div>

            {mode === 'STREAM' ? (
                /* LIVE STREAM VIEW */
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1">
                    <div className="lg:col-span-2 space-y-6">
                        <div className="admin-card rounded-2xl p-6 bg-gradient-to-br from-gray-800 to-gray-900">
                            <div className="flex justify-between items-center mb-6">
                                <div className="flex items-center gap-2 text-gray-300">
                                    <Activity size={18} className="text-green-400" />
                                    <span className="font-semibold tracking-wide uppercase text-sm">Vital Signs</span>
                                </div>
                                <div className="flex items-center gap-3">
                                    <span className="text-xs font-mono text-gray-500">IOT-7752</span>
                                    <button onClick={() => setIsStreaming(!isStreaming)} className="text-gray-400 hover:text-white transition-colors">
                                        {isStreaming ? <Pause size={18} /> : <Play size={18} />}
                                    </button>
                                </div>
                            </div>

                            <div className="grid grid-cols-2 gap-6">
                                <VitalCard
                                    icon={<Heart />} label="Heart Rate" value={vitals.hr} unit="bpm"
                                    history={vitals.hrHistory} color="#ef4444" warning={vitals.hr > 100 || vitals.hr < 55}
                                />
                                <VitalCard
                                    icon={<span className="font-bold text-sm">O₂</span>} label="SpO2" value={vitals.spo2} unit="%"
                                    history={vitals.spo2History} color="#22d3d1" warning={vitals.spo2 < 94}
                                />
                                <VitalCard
                                    icon={<Thermometer />} label="Temp" value={vitals.temp} unit="°C"
                                    history={vitals.tempHistory} color="#f59e0b" warning={vitals.temp > 38}
                                />
                                <VitalCard
                                    icon={<Footprints />} label="Steps" value={vitals.steps} unit="/min"
                                    history={vitals.stepsHistory} color="#10b981"
                                />
                            </div>
                        </div>
                    </div>

                    <div className="space-y-6">
                        {/* AI Prediction Panel */}
                        <div className="admin-card rounded-2xl p-8 text-center flex flex-col items-center justify-center h-full relative overflow-hidden">
                            <div className="absolute inset-0 bg-blue-500/5"></div>
                            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-blue-500 to-transparent opacity-50"></div>

                            <h3 className="text-gray-400 text-xs font-bold uppercase tracking-widest mb-6 z-10">Real-time Risk Analysis</h3>

                            <div className={`w-40 h-40 rounded-full flex items-center justify-center border-[6px] transition-all duration-500 z-10 ${streamPrediction?.color === 'red' ? 'border-red-500 shadow-[0_0_30px_rgba(239,68,68,0.4)]' :
                                    streamPrediction?.color === 'yellow' ? 'border-yellow-500 shadow-[0_0_30px_rgba(245,158,11,0.4)]' :
                                        'border-green-500 shadow-[0_0_30px_rgba(16,185,129,0.4)]'
                                }`}>
                                <div className="text-center">
                                    <span className={`text-4xl font-bold ${streamPrediction?.color === 'red' ? 'text-red-400' :
                                            streamPrediction?.color === 'yellow' ? 'text-yellow-400' :
                                                'text-green-400'
                                        }`}>
                                        {Math.round((streamPrediction?.risk || 0) * 100)}%
                                    </span>
                                    <div className="text-[10px] text-gray-400 mt-1 uppercase">Probability</div>
                                </div>
                            </div>

                            <div className={`mt-6 text-2xl font-bold tracking-tight z-10 ${streamPrediction?.color === 'red' ? 'text-red-400' :
                                    streamPrediction?.color === 'yellow' ? 'text-yellow-400' :
                                        'text-green-400'
                                }`}>
                                {streamPrediction?.level || 'Initializing...'}
                            </div>

                            <div className="mt-8 w-full bg-black/20 rounded-lg p-3 text-left font-mono text-xs text-blue-300 border border-white/5 z-10">
                                <div>&gt; Model: MLP-Fusion-v2</div>
                                <div>&gt; Update: {(new Date()).toLocaleTimeString()}</div>
                                <div>&gt; Status: <span className="text-green-400">Active</span></div>
                            </div>
                        </div>
                    </div>
                </div>
            ) : (
                /* MANUAL ENTRY VIEW */
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 flex-1 items-start">
                    <div className="admin-card rounded-2xl p-8">
                        <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                            <Keyboard className="text-purple-400" /> Manual Clinical Entry
                        </h2>

                        <div className="space-y-6">
                            <InputSlider
                                label="Heart Rate (BPM)" icon={<Heart size={16} />}
                                value={manualInputs.heart_rate}
                                onChange={v => setManualInputs({ ...manualInputs, heart_rate: v })}
                                min={30} max={200} color="red"
                            />
                            <InputSlider
                                label="O2 Saturation (%)" icon={<Wind size={16} />}
                                value={manualInputs.oxygen_saturation}
                                onChange={v => setManualInputs({ ...manualInputs, oxygen_saturation: v })}
                                min={70} max={100} color="cyan"
                            />
                            <div className="grid grid-cols-2 gap-4">
                                <InputBox
                                    label="Sys BP" value={manualInputs.blood_pressure_sys}
                                    onChange={v => setManualInputs({ ...manualInputs, blood_pressure_sys: Number(v) })}
                                />
                                <InputBox
                                    label="Dia BP" value={manualInputs.blood_pressure_dia}
                                    onChange={v => setManualInputs({ ...manualInputs, blood_pressure_dia: Number(v) })}
                                />
                            </div>

                            <button
                                onClick={handleManualAnalyze}
                                disabled={isAnalyzing}
                                className="w-full py-4 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 text-white rounded-xl font-bold text-lg shadow-xl shadow-purple-500/20 transition-all flex items-center justify-center gap-2"
                            >
                                {isAnalyzing ? <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" /> : <Zap />}
                                Run Inference
                            </button>
                            <p className="text-xs text-center text-gray-500">Processed by Hospital B Local Model (Privacy Preserved)</p>
                        </div>
                    </div>

                    <div className="flex flex-col gap-6">
                        {manualResult ? (
                            <div className="admin-card rounded-2xl p-8 border-l-4 border-l-purple-500 animate-in fade-in slide-in-from-bottom-4 bg-gradient-to-br from-gray-800 to-gray-900">
                                <h3 className="text-gray-400 text-sm font-bold uppercase tracking-widest mb-4">Inference Result</h3>

                                <div className="flex items-center justify-between mb-8">
                                    <div>
                                        <div className="text-5xl font-bold text-white mb-1">
                                            {(manualResult.risk_score * 100).toFixed(1)}%
                                        </div>
                                        <div className="text-sm text-gray-400">Risk Probability</div>
                                    </div>
                                    <div className={`px-4 py-2 rounded-lg font-bold text-lg border ${manualResult.risk_level.includes('Critical') ? 'bg-red-500/10 border-red-500/50 text-red-400' :
                                            manualResult.risk_level.includes('Warning') ? 'bg-yellow-500/10 border-yellow-500/50 text-yellow-400' :
                                                'bg-green-500/10 border-green-500/50 text-green-400'
                                        }`}>
                                        {manualResult.risk_level}
                                    </div>
                                </div>

                                <div className="space-y-3">
                                    <FactorRow label="HR Impact" val={manualInputs.heart_rate > 100 || manualInputs.heart_rate < 55 ? 'High' : 'Normal'} />
                                    <FactorRow label="SpO2 Impact" val={manualInputs.oxygen_saturation < 94 ? 'Hypoxia Risk' : 'Normal'} />
                                    <FactorRow label="Model Confidence" val="94.2%" />
                                </div>
                            </div>
                        ) : (
                            <div className="admin-card rounded-2xl p-12 text-center border-dashed border-gray-700 flex flex-col items-center justify-center h-64 text-gray-500">
                                <Activity size={48} className="mb-4 opacity-20" />
                                <p>Enter vitals and run inference to see AI prediction</p>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}

function VitalCard({ icon, label, value, unit, history, color, warning }) {
    return (
        <div className="bg-black/20 rounded-xl p-4 border border-white/5 relative overflow-hidden group hover:border-white/10 transition-colors">
            <div className={`absolute top-0 left-0 w-1 h-full`} style={{ backgroundColor: warning ? '#ef4444' : color }}></div>
            <div className="flex justify-between items-start mb-2 pl-3">
                <div>
                    <div className="text-gray-400 text-xs font-semibold uppercase">{label}</div>
                    <div className={`text-2xl font-bold font-mono mt-1 ${warning ? 'text-red-400 animate-pulse' : 'text-white'}`}>
                        {value} <span className="text-xs font-sans font-normal text-gray-500 ml-1">{unit}</span>
                    </div>
                </div>
                <div className={`p-2 rounded-lg bg-white/5 text-gray-300`}>
                    {icon}
                </div>
            </div>

            {/* Sparkline */}
            <div className="h-12 w-full pl-3 opacity-50 group-hover:opacity-100 transition-opacity">
                <svg viewBox="0 0 160 40" className="w-full h-full overflow-visible">
                    <defs>
                        <linearGradient id={`grad-${label}`} x1="0" x2="0" y1="0" y2="1">
                            <stop offset="0%" stopColor={color} stopOpacity="0.5" />
                            <stop offset="100%" stopColor={color} stopOpacity="0" />
                        </linearGradient>
                    </defs>
                    <path
                        d={`M ${history.map((v, i) => {
                            const min = Math.min(...history) * 0.95;
                            const max = Math.max(...history) * 1.05;
                            const range = max - min || 1;
                            const x = (i / (history.length - 1)) * 160;
                            const y = 40 - ((v - min) / range) * 40;
                            return `${x},${y}`;
                        }).join(' ')}`}
                        fill="none"
                        stroke={color}
                        strokeWidth="2"
                        strokeLinecap="round"
                        vectorEffect="non-scaling-stroke"
                    />
                </svg>
            </div>
        </div>
    );
}

function InputSlider({ label, icon, value, onChange, min, max, color }) {
    return (
        <div className="bg-black/20 p-4 rounded-xl border border-white/5">
            <div className="flex justify-between mb-2">
                <div className="flex items-center gap-2 text-gray-300 font-medium text-sm">
                    <span className={`text-${color}-400`}>{icon}</span> {label}
                </div>
                <div className="font-mono text-white font-bold">{value}</div>
            </div>
            <input
                type="range" min={min} max={max} value={value}
                onChange={e => onChange(Number(e.target.value))}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
            />
        </div>
    )
}

function InputBox({ label, value, onChange }) {
    return (
        <div className="bg-black/20 p-4 rounded-xl border border-white/5">
            <label className="text-gray-400 text-xs font-bold uppercase block mb-2">{label}</label>
            <input
                type="number" value={value} onChange={e => onChange(e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white font-mono focus:outline-none focus:border-blue-500 transition-colors"
            />
        </div>
    )
}

function FactorRow({ label, val }) {
    return (
        <div className="flex justify-between items-center py-2 border-b border-gray-700 last:border-0">
            <span className="text-gray-400 text-sm">{label}</span>
            <span className={`text-sm font-bold ${val === 'High' || val.includes('Risk') ? 'text-red-400' : 'text-green-400'
                }`}>{val}</span>
        </div>
    )
}
