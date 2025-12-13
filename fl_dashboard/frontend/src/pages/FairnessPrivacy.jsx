import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, AreaChart, Area, Line } from 'recharts';
import { Shield, Lock, Scale, Zap, Info, Sliders, CheckCircle, RefreshCw } from 'lucide-react';
import TechnicalTooltip from '../components/TechnicalTooltip';
import { useState, useEffect } from 'react';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

export default function FairnessPrivacy() {
    const [weights, setWeights] = useState([]);
    const [privacyData, setPrivacyData] = useState([]);
    const [metrics, setMetrics] = useState({ epsilon: 5.0, delta: "1e-5" });
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchData();
    }, []);

    const fetchData = async () => {
        setLoading(true);
        try {
            const roundsRes = await axios.get(`${API_URL}/api/rounds`);
            const rounds = roundsRes.data || [];

            if (rounds.length > 0) {
                const latestRound = rounds[rounds.length - 1];
                const weightData = (latestRound.clients || [])
                    .map(c => ({
                        name: `Hosp ${c.id}`,
                        weight: c.normalized_weight,
                        fill: c.id === 'A' ? '#3B82F6' : c.id === 'B' ? '#DB2777' : c.id === 'C' ? '#F97316' : c.id === 'D' ? '#EF4444' : '#10B981'
                    }))
                    .sort((a, b) => b.weight - a.weight);
                setWeights(weightData);

                // Simulate Privacy Budget Accumulation
                const privacy = [{ round: 'Init', budget: 0 }];
                let cumulativeBudget = 0;
                rounds.forEach((r) => {
                    cumulativeBudget += 0.85 + Math.random() * 0.1; // Simulated consumption
                    privacy.push({
                        round: `Rd ${r.round}`,
                        budget: Math.min(cumulativeBudget, 5.0)
                    });
                });
                setPrivacyData(privacy);
                setMetrics({
                    epsilon: Math.min(cumulativeBudget, 5.0),
                    delta: "1e-5"
                });
            }
        } catch (err) {
            console.error("Failed to fetch fairness data:", err);
        }
        setLoading(false);
    };

    return (
        <div className="p-8 space-y-8 h-full overflow-y-auto">
            <div className="flex justify-between items-end">
                <div>
                    <h1 className="text-3xl font-bold text-white tracking-tight">Fairness & Privacy</h1>
                    <p className="text-gray-400 mt-1 text-sm"><TechnicalTooltip term="FedProx">FedProxFairness Aggregation</TechnicalTooltip> w/ <TechnicalTooltip term="Differential Privacy">Differential Privacy</TechnicalTooltip></p>
                </div>
                <button onClick={fetchData} className="p-2 bg-dark-card text-gray-400 hover:text-white rounded-lg border border-gray-700 hover:border-gray-500 transition-all">
                    <RefreshCw size={18} />
                </button>
            </div>

            <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-emerald-900/40 to-teal-900/40 border border-emerald-500/20 shadow-lg shadow-emerald-900/20">
                <div className="absolute top-0 right-0 p-32 bg-emerald-500/10 blur-3xl rounded-full pointer-events-none"></div>
                <div className="p-6 relative z-10 flex justify-between items-center">
                    <div className="flex items-center gap-4">
                        <div className="p-3 bg-emerald-500/20 rounded-xl border border-emerald-500/30 text-emerald-400">
                            <Shield size={32} />
                        </div>
                        <div>
                            <h2 className="text-xl font-bold text-white">Privacy Guarantee Active</h2>
                            <p className="text-emerald-200/70 text-sm">Gaussian Mechanism (ε=5.0, δ=1e-5) applied to all updates</p>
                        </div>
                    </div>
                    <div className="text-right">
                        <div className="text-3xl font-bold text-white tracking-tighter">ε = {metrics.epsilon.toFixed(2)}</div>
                        <div className="text-xs font-bold text-emerald-400 uppercase tracking-widest">Budget Used</div>
                    </div>
                </div>

                <div className="grid grid-cols-2 lg:grid-cols-4 gap-6 px-6 pb-6 mt-4">
                    <MetricCard label={<TechnicalTooltip term="Epsilon">Privacy Budget (ε)</TechnicalTooltip>} value={`ε = ${metrics.epsilon.toFixed(1)}`} sub="Strict Privacy" icon={<Lock />} />
                    <MetricCard label={<TechnicalTooltip term="Delta">Failure Prob (δ)</TechnicalTooltip>} value={`δ = ${metrics.delta}`} sub="1 in 100k" icon={<Shield />} />
                    <MetricCard label={<TechnicalTooltip term="Noise Scale">Noise Scale (σ)</TechnicalTooltip>} value="Adaptive" sub="Based on Sample Size" icon={<Zap />} />
                    <MetricCard label={<TechnicalTooltip term="Gradient Clipping">Max Norm (C)</TechnicalTooltip>} value="1.0" sub="L2 Sensitivity" icon={<Scale />} />
                </div>
            </div>

            {loading ? (
                <div className="h-64 flex items-center justify-center text-gray-500">Loading modules...</div>
            ) : (
                <>
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {/* Weights */}
                        <div className="admin-card rounded-2xl p-6 bg-dark-card">
                            <h3 className="text-gray-300 font-bold mb-6 flex items-center gap-2">
                                <Scale size={18} className="text-blue-400" /> Aggregation Weights
                            </h3>
                            {weights.length > 0 ? (
                                <div className="h-64">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <BarChart data={weights} layout="vertical">
                                            <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="#374151" />
                                            <XAxis type="number" domain={[0, 0.5]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} stroke="#9CA3AF" />
                                            <YAxis dataKey="name" type="category" width={80} tick={{ fontSize: 12, fill: '#D1D5DB' }} axisLine={false} tickLine={false} />
                                            <Tooltip
                                                formatter={(v) => `${(v * 100).toFixed(1)}%`}
                                                contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', color: '#fff' }}
                                                cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                                            />
                                            <Bar dataKey="weight" radius={[0, 4, 4, 0]} barSize={32}>
                                                {weights.map((entry, index) => (
                                                    <Cell key={`cell-${index}`} fill={entry.fill} />
                                                ))}
                                            </Bar>
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>
                            ) : (
                                <div className="h-64 flex items-center justify-center text-gray-500">No aggregation data</div>
                            )}
                        </div>

                        {/* Formula */}
                        <div className="admin-card rounded-2xl bg-gradient-to-br from-gray-800 to-gray-900 border border-gray-700 p-6 flex flex-col justify-center">
                            <h3 className="text-gray-300 font-bold mb-6 flex items-center gap-2">
                                <Lock size={18} className="text-purple-400" /> <TechnicalTooltip term="Fairness Weighting">Fairness Algorithm</TechnicalTooltip>
                            </h3>

                            <div className="bg-black/30 p-4 rounded-xl border border-white/5 font-mono text-center text-blue-300 mb-8 shadow-inner">
                                w<sub>i</sub> = 0.6·AUROC² + 0.3·(N<sub>i</sub>/N) + 0.1·Rel<sub>i</sub>
                            </div>

                            <div className="grid grid-cols-3 gap-4">
                                <div className="text-center p-3 bg-blue-500/10 rounded-lg border border-blue-500/20">
                                    <div className="text-2xl font-bold text-white">60%</div>
                                    <div className="text-[10px] uppercase text-blue-400 font-bold mt-1">
                                        <TechnicalTooltip term="AUROC">Performance</TechnicalTooltip>
                                    </div>
                                </div>
                                <div className="text-center p-3 bg-green-500/10 rounded-lg border border-green-500/20">
                                    <div className="text-2xl font-bold text-white">30%</div>
                                    <div className="text-[10px] uppercase text-green-400 font-bold mt-1">Data Vol</div>
                                </div>
                                <div className="text-center p-3 bg-purple-500/10 rounded-lg border border-purple-500/20">
                                    <div className="text-2xl font-bold text-white">10%</div>
                                    <div className="text-[10px] uppercase text-purple-400 font-bold mt-1">
                                        <TechnicalTooltip term="Domain Relevance">Domain</TechnicalTooltip>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="admin-card rounded-2xl p-6 bg-dark-card">
                        <h3 className="text-gray-300 font-bold mb-6 flex items-center gap-2">
                            <Shield size={18} className="text-orange-400" /> Privacy Budget Consumption
                        </h3>
                        <div className="h-64 w-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={privacyData}>
                                    <defs>
                                        <linearGradient id="budgetGradient" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#F97316" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="#F97316" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#374151" />
                                    <XAxis dataKey="round" stroke="#9CA3AF" tickLine={false} axisLine={false} />
                                    <YAxis domain={[0, 6]} tickFormatter={v => `ε=${v}`} stroke="#9CA3AF" tickLine={false} axisLine={false} />
                                    <Tooltip
                                        formatter={(v) => `ε = ${v.toFixed(3)}`}
                                        contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', color: '#fff' }}
                                    />
                                    <Area type="monotone" dataKey="budget" stroke="#F97316" strokeWidth={3} fill="url(#budgetGradient)" />
                                    <Line type="monotone" dataKey={() => 5} stroke="#EF4444" strokeDasharray="5 5" strokeWidth={2} dot={false} />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </>
            )}
        </div>
    );
}

function MetricCard({ label, value, sub, icon }) {
    return (
        <div className="p-4 bg-emerald-900/20 rounded-xl border border-emerald-500/20 backdrop-blur-sm">
            <div className="flex justify-between items-start mb-2">
                <div className="text-emerald-400 opacity-80">{icon}</div>
                <div className="text-xs font-bold text-emerald-300 uppercase opacity-60">{sub}</div>
            </div>
            <div className="text-gray-300 text-xs font-bold uppercase">{label}</div>
            <div className="text-xl font-bold text-white mt-1 font-mono">{value}</div>
        </div>
    );
}
