import { useState } from 'react';
import { Database, Server, Activity, Cpu, HardDrive, BarChart3, Layers, CheckCircle2, Upload, Calendar, FileText, Lock } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

export default function HospitalProfiles() {
    const [activeTab, setActiveTab] = useState('A');
    const [showUploadModal, setShowUploadModal] = useState(false);

    const hospitals = {
        A: {
            id: 'A',
            name: 'Cardiology Center A',
            type: 'Specialist Node',
            modality: '12-Lead ECG',
            samples: 19601,
            tech: 'S4 Model (36L, 5.6M Params)',
            description: 'Major regional cardiac center contributing high-volume ECG data from PTB-XL. Uses State Space Models (S4) for long-range dependency modeling.',
            stats: [
                { label: 'Classes', value: '5 (NORM, MI, STTC...)' },
                { label: 'Leads', value: '8-12 Channel' },
                { label: 'Architecture', value: 'S4 Encoder + MLP' }
            ],
            performance: [
                { stage: 'Baseline', value: 0.75 },
                { stage: 'Federated', value: 0.82 },
                { stage: 'Gain', value: 0.07 }
            ],
            details: {
                training: '10 epochs, 3 hours (CPU)',
                privacy: 'σ = 0.000081 (Adaptive Noise)',
                contribution: '85.3% of FL Weight (High Volume)'
            }
        },
        B: {
            id: 'B',
            name: 'General Hospital B',
            type: 'Clinical Node',
            modality: 'Tabular Vitals',
            samples: 1000,
            tech: 'Lightweight MLP (3K Params)',
            description: 'Community hospital providing critical care vitals (SpO2, BP, HR). Achieves near-perfect local accuracy for deterioration prediction.',
            stats: [
                { label: 'Features', value: '15 Clinical Marks' },
                { label: 'Recall', value: '1.0 (High Sensitivity)' },
                { label: 'Top Feature', value: 'SpO2 Saturation' }
            ],
            performance: [
                { stage: 'Baseline', value: 0.959 },
                { stage: 'Federated', value: 0.98 },
                { stage: 'Gain', value: 0.021 }
            ],
            details: {
                training: '10 epochs, 2 minutes',
                privacy: 'σ = 0.001764 (High Noise/Sample)',
                contribution: '26% Weight (Quality Boost)'
            }
        },
        C: {
            id: 'C',
            name: 'Imaging Center C',
            type: 'Radiology Node',
            modality: 'Chest X-Ray',
            samples: 200,
            tech: 'ResNet50 (25.6M Params)',
            description: 'Specialized imaging partner. Small dataset but high-dimensional RGB data (224x224). Benefits significantly from FL regularization.',
            stats: [
                { label: 'Resolution', value: '224x224 RGB' },
                { label: 'Backbone', value: 'ResNet50 Pretrained' },
                { label: 'Augmentation', value: 'RandAugment' }
            ],
            performance: [
                { stage: 'Baseline', value: 0.65 },
                { stage: 'Federated', value: 0.69 },
                { stage: 'Gain', value: 0.04 }
            ],
            details: {
                training: '3 epochs, 10 minutes',
                privacy: 'σ = 0.0088 (Max Privacy Cost)',
                contribution: '12.8% Weight'
            }
        },
        D: {
            id: 'D',
            name: 'Geriatric Clinic D',
            type: 'Specialist (Age ≥60)',
            modality: 'Geriatric ECG',
            samples: 3000,
            tech: 'S4-Lite (18M Params)',
            description: 'Clinic focused on elderly care. Data generated using diffusion models to simulate significant distribution shift (Age ≥60). Biggest beneficiary of Federation.',
            stats: [
                { label: 'Population', value: 'Age ≥ 60' },
                { label: 'Data Source', value: 'Synthetic (Diffusion)' },
                { label: 'Shift', value: 'Non-IID Distribution' }
            ],
            performance: [
                { stage: 'Baseline', value: 0.68 },
                { stage: 'Federated', value: 0.78 },
                { stage: 'Gain', value: 0.10 }
            ],
            details: {
                training: '5 epochs, 4-6 hours',
                privacy: 'σ = 0.000588',
                contribution: '16.0% Weight'
            }
        },
        E: {
            id: 'E',
            name: 'Research Institute E',
            type: 'Multimodal Hub',
            modality: 'ECG + Vitals + X-Ray',
            samples: 3000,
            tech: 'Fusion Transformer (5.2M)',
            description: 'Advanced research node fusing multiple data streams. Bridges the domain gap between disparate hospitals.',
            stats: [
                { label: 'Fusion', value: 'Late-Stage Attention' },
                { label: 'Inputs', value: '3 Modalities' },
                { label: 'Role', value: 'Domain Bridge' }
            ],
            performance: [
                { stage: 'Baseline', value: 0.78 },
                { stage: 'Federated', value: 0.85 },
                { stage: 'Gain', value: 0.07 }
            ],
            details: {
                training: '5 epochs, 6-8 hours',
                privacy: 'σ = 0.000588',
                contribution: '18.9% Weight (Bridge)'
            }
        }
    };

    const active = hospitals[activeTab];

    return (
        <div className="p-8 space-y-8 h-full overflow-y-auto relative">
            <div className="flex justify-between items-end">
                <div>
                    <h1 className="text-3xl font-bold text-white tracking-tight">Hospital Profiles</h1>
                    <p className="text-gray-400 mt-1 text-sm">Node Configuration & Local Performance Metadata</p>
                </div>
                <button
                    onClick={() => setShowUploadModal(true)}
                    className="flex items-center gap-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-bold text-sm shadow-lg shadow-blue-500/20 transition-all"
                >
                    <Upload size={18} /> Upload Monthly Data
                </button>
            </div>

            <div className="grid grid-cols-12 gap-8">
                {/* Sidebar Selector */}
                <div className="col-span-12 lg:col-span-3 space-y-2">
                    {Object.values(hospitals).map((h) => (
                        <button
                            key={h.id}
                            onClick={() => setActiveTab(h.id)}
                            className={`w-full text-left p-4 rounded-xl border transition-all duration-200 group relative overflow-hidden ${activeTab === h.id
                                ? 'bg-blue-600 border-blue-500 shadow-lg shadow-blue-900/40'
                                : 'bg-dark-card border-transparent hover:bg-white/5 text-gray-400 hover:text-white'
                                }`}
                        >
                            <div className="relative z-10 flex items-center justify-between">
                                <div>
                                    <div className={`font-bold text-lg ${activeTab === h.id ? 'text-white' : ''}`}>Hospital {h.id}</div>
                                    <div className={`text-xs ${activeTab === h.id ? 'text-blue-200' : 'text-gray-500'}`}>{h.modality}</div>
                                </div>
                                <div className={`p-2 rounded-lg ${activeTab === h.id ? 'bg-white/20' : 'bg-black/20'}`}>
                                    {h.id === 'A' || h.id === 'D' ? <Activity size={18} /> :
                                        h.id === 'C' ? <Layers size={18} /> : <Database size={18} />}
                                </div>
                            </div>
                        </button>
                    ))}
                </div>

                {/* content */}
                <div className="col-span-12 lg:col-span-9 space-y-6">
                    <div className="admin-card rounded-2xl p-8 bg-gradient-to-br from-gray-800 to-gray-900 relative border border-white/5">
                        <div className="flex justify-between items-start mb-6">
                            <div>
                                <div className="flex items-center gap-3 mb-2">
                                    <h2 className="text-2xl font-bold text-white">{active.name}</h2>
                                    <span className="px-2 py-1 rounded-md bg-blue-500/20 border border-blue-500/30 text-xs font-bold text-blue-300 uppercase">{active.type}</span>
                                </div>
                                <p className="text-gray-400 max-w-xl text-sm leading-relaxed">{active.description}</p>
                            </div>
                            <div className="text-right">
                                <div className="text-3xl font-bold text-white">{active.samples.toLocaleString()}</div>
                                <div className="text-xs font-bold text-gray-500 uppercase tracking-wider">Total Samples</div>
                            </div>
                        </div>

                        <div className="grid grid-cols-3 gap-4 mb-8">
                            {active.stats.map((s, i) => (
                                <div key={i} className="p-4 bg-black/20 rounded-xl border border-white/5 hover:border-white/10 transition-colors">
                                    <div className="text-xs text-gray-500 uppercase font-bold mb-1">{s.label}</div>
                                    <div className="text-sm font-mono font-bold text-blue-300">{s.value}</div>
                                </div>
                            ))}
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                            {/* Performance Chart */}
                            <div className="h-64">
                                <h3 className="text-xs font-bold text-gray-400 uppercase mb-4 flex items-center gap-2">
                                    <BarChart3 size={16} /> FL Impact Analysis
                                </h3>
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={active.performance} layout="vertical" barSize={24}>
                                        <CartesianGrid strokeDasharray="3 3" horizontal={false} vertical={true} stroke="#374151" />
                                        <XAxis type="number" domain={[0, 1]} hide />
                                        <YAxis dataKey="stage" type="category" width={80} tick={{ fill: '#9CA3AF', fontSize: 11, fontWeight: 600 }} axisLine={false} tickLine={false} />
                                        <Tooltip
                                            cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                                            contentStyle={{ backgroundColor: '#1f2937', color: '#fff', border: '1px solid #374151', borderRadius: '8px' }}
                                        />
                                        <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                                            {active.performance.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={
                                                    index === 0 ? '#4B5563' :
                                                        index === 1 ? '#3B82F6' : '#10B981'
                                                } />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>

                            {/* Technical Details */}
                            <div className="space-y-4">
                                <h3 className="text-xs font-bold text-gray-400 uppercase mb-4 flex items-center gap-2">
                                    <Cpu size={16} /> Technical Specifications
                                </h3>
                                <DetailRow label="Model Arch" value={active.tech} />
                                <DetailRow label="Training Config" value={active.details.training} />
                                <DetailRow label="Privacy Budget" value={active.details.privacy} />
                                <DetailRow label="Net Contribution" value={active.details.contribution} highlight />
                            </div>
                        </div>
                    </div>

                    <div className="grid grid-cols-3 gap-6">
                        <StatusCard icon={<Server />} label="Endpoint" value="Active (Port 8000)" color="green" />
                        <StatusCard icon={<Database />} label="Storage" value="Encrypted (AES-256)" color="blue" />
                        <StatusCard icon={<HardDrive />} label="Last Sync" value="12 Dec 2025" color="purple" />
                    </div>
                </div>
            </div>

            {/* Upload Modal */}
            {showUploadModal && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4 animate-in fade-in duration-200">
                    <div className="bg-[#1e2129] w-full max-w-md rounded-2xl border border-gray-700 shadow-2xl overflow-hidden animate-in zoom-in-95 duration-200">
                        <div className="p-6 border-b border-gray-700 bg-gradient-to-r from-blue-900/20 to-purple-900/20">
                            <h2 className="text-xl font-bold text-white flex items-center gap-2">
                                <Calendar className="text-blue-400" /> Monthly Data Upload
                            </h2>
                            <p className="text-gray-400 text-sm mt-1">Securely ingest new patient records</p>
                        </div>

                        <div className="p-6 space-y-6">
                            <div className="space-y-2">
                                <label className="text-xs font-bold text-gray-400 uppercase">Select Quarter</label>
                                <select className="w-full bg-black/20 border border-gray-600 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-blue-500">
                                    <option>Q1 2026 (January)</option>
                                    <option>Q1 2026 (February)</option>
                                    <option>Q1 2026 (March)</option>
                                </select>
                            </div>

                            <div className="border-2 border-dashed border-gray-600 rounded-xl p-8 flex flex-col items-center justify-center text-center hover:bg-white/5 transition-colors cursor-pointer group">
                                <div className="w-12 h-12 bg-gray-700 rounded-full flex items-center justify-center mb-3 group-hover:bg-blue-600 transition-colors">
                                    <FileText className="text-gray-300 group-hover:text-white" />
                                </div>
                                <div className="text-sm font-bold text-gray-300">Drop CSV or Parquet files</div>
                                <div className="text-xs text-gray-500 mt-1">Maximum size: 500MB</div>
                            </div>

                            <div className="flex items-center gap-2 text-xs text-gray-500 bg-blue-900/10 p-3 rounded-lg border border-blue-900/30">
                                <Lock size={12} className="text-blue-400" />
                                Data is automatically anonymized before ingestion.
                            </div>
                        </div>

                        <div className="p-4 border-t border-gray-700 flex justify-end gap-3 bg-black/10">
                            <button
                                onClick={() => setShowUploadModal(false)}
                                className="px-4 py-2 rounded-lg text-sm font-medium text-gray-400 hover:text-white hover:bg-white/5 transition-all"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={() => {
                                    // Mock upload
                                    setTimeout(() => setShowUploadModal(false), 1000);
                                }}
                                className="px-4 py-2 rounded-lg text-sm font-bold bg-blue-600 text-white hover:bg-blue-500 shadow-lg shadow-blue-500/20 transition-all flex items-center gap-2"
                            >
                                <Upload size={14} /> Upload Records
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

function DetailRow({ label, value, highlight }) {
    return (
        <div className="flex justify-between items-center py-2 border-b border-gray-700 last:border-0">
            <span className="text-gray-500 text-xs font-bold uppercase">{label}</span>
            <span className={`text-sm font-mono ${highlight ? 'text-green-400 font-bold' : 'text-gray-300'}`}>{value}</span>
        </div>
    )
}

function StatusCard({ icon, label, value, color }) {
    const colors = {
        green: 'bg-green-500/10 text-green-400 border-green-500/20',
        blue: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
        purple: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
    }
    return (
        <div className={`p-4 rounded-xl border flex items-center gap-3 ${colors[color]}`}>
            <div className="p-2 bg-black/10 rounded-lg">{icon}</div>
            <div>
                <div className="text-[10px] font-bold uppercase opacity-70">{label}</div>
                <div className="text-sm font-bold">{value}</div>
            </div>
        </div>
    )
}
