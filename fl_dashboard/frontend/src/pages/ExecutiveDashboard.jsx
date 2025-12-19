
import { Download, Users, Activity, Shield, Database, Box, TrendingUp, TrendingDown, RefreshCw } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import TechnicalTooltip from '../components/TechnicalTooltip';
import { useState, useEffect } from 'react';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

export default function ExecutiveDashboard() {
    const [status, setStatus] = useState(null);
    const [metrics, setMetrics] = useState(null);
    const [rounds, setRounds] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchData();
    }, []);

    const fetchData = async () => {
        setLoading(true);
        try {
            const [statusRes, metricsRes, roundsRes] = await Promise.all([
                axios.get(`${API_URL}/api/status`),
                axios.get(`${API_URL}/api/metrics`),
                axios.get(`${API_URL}/api/rounds`)
            ]);
            setStatus(statusRes.data);
            setMetrics(metricsRes.data);
            setRounds(roundsRes.data || []);
        } catch (err) {
            console.error("Failed to fetch dashboard data:", err);
        }
        setLoading(false);
    };

    const performanceData = (() => {
        if (!metrics) return [];
        return ['A', 'B', 'C', 'D', 'E'].map(h => ({
            hospital: h,
            baseline: metrics.before_fl?.[h] || 0.5,
            afterFL: metrics.after_fl?.[h] || 0.5,
            personalized: metrics.after_personalization?.[h] || 0.5
        }));
    })();

    const avgBaseline = performanceData.reduce((a, b) => a + b.baseline, 0) / 5;
    const avgPers = performanceData.reduce((a, b) => a + b.personalized, 0) / 5;

    return (
        <div className="p-8 space-y-8 h-full overflow-y-auto">
            {/* Header */}
            <div className="flex justify-between items-end">
                <div>
                    <h1 className="text-3xl font-bold text-white tracking-tight">Executive Dashboard</h1>
                    <p className="text-gray-400 mt-1 text-sm bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent font-medium">
                        Network Status: Active • Round {status?.round_current || 3}
                    </p>
                </div>

                <div className="flex gap-3">
                    <button onClick={fetchData} className="p-2.5 bg-dark-card border border-gray-700 text-gray-400 rounded-lg hover:text-white hover:border-gray-600 transition-all">
                        <RefreshCw size={18} />
                    </button>
                    <a href={`${API_URL} /api/download / all`} className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white px-5 py-2.5 rounded-lg font-bold text-sm shadow-lg shadow-blue-500/20 transition-all">
                        <Download size={18} /> Download Report
                    </a>
                </div>
            </div>

            {loading ? (
                <div className="h-64 flex items-center justify-center text-gray-500">Loading analytics...</div>
            ) : (
                <>
                    {/* KPI Grid */}
                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                        <KpiCard icon={<Users />} label="Hospitals" value={status?.hospitals || 5} sub="Active Nodes" color="text-blue-400" bg="bg-blue-500/10" />
                        <KpiCard icon={<Activity />} label="FL Rounds" value={status?.round_current || 3} sub="Completed" color="text-purple-400" bg="bg-purple-500/10" />
                        <KpiCard icon={<Database />} label="Samples" value={(status?.total_samples || 27018).toLocaleString()} sub="Total Training" color="text-green-400" bg="bg-green-500/10" />
                        <KpiCard icon={<Shield />} label="Privacy Bal" value={`ε = ${status?.privacy_budget || 5.0} `} sub="Budget Remaining" color="text-orange-400" bg="bg-orange-500/10" />
                        <KpiCard icon={<Box />} label="Ledger" value={status?.blockchain_blocks || 60} sub="Verified Blocks" color="text-gray-400" bg="bg-gray-500/10" />
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                        {/* Chart */}
                        <div className="lg:col-span-2 admin-card rounded-2xl p-6 bg-dark-card">
                            <h3 className="text-gray-200 font-bold mb-6 flex items-center justify-between">
                                <span>Performance Trajectory</span>
                                <span className="text-xs font-normal text-gray-500 bg-black/20 px-2 py-1 rounded">AUROC Metric</span>
                            </h3>
                            <div className="h-72">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={performanceData} barGap={4}>
                                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#374151" />
                                        <XAxis dataKey="hospital" tickFormatter={v => `Hosp ${v} `} stroke="#9CA3AF" tickLine={false} axisLine={false} />
                                        <YAxis domain={[0.4, 0.9]} stroke="#9CA3AF" tickLine={false} axisLine={false} tickFormatter={v => v.toFixed(1)} />
                                        <Tooltip
                                            contentStyle={{ backgroundColor: '#1f2937', borderColor: '#374151', color: '#fff' }}
                                            itemStyle={{ color: '#fff' }}
                                            cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                                        />
                                        <Bar dataKey="baseline" name="Baseline" fill="#4B5563" radius={[4, 4, 0, 0]} />
                                        <Bar dataKey="afterFL" name="After FL" fill="#3B82F6" radius={[4, 4, 0, 0]} />
                                        <Bar dataKey="personalized" name="Personalized" fill="#10B981" radius={[4, 4, 0, 0]} />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        {/* Stats Panel */}
                        <div className="admin-card rounded-2xl p-0 overflow-hidden flex flex-col">
                            <div className="p-6 bg-gradient-to-br from-blue-900/20 to-purple-900/20 border-b border-white/5">
                                <h3 className="text-gray-200 font-bold mb-4">Gains Summary</h3>
                                <div className="flex justify-between items-end mb-2">
                                    <span className="text-gray-400 text-sm">Avg Improvement</span>
                                    <span className="text-green-400 font-bold text-xl flex items-center gap-1">
                                        <TrendingUp size={16} /> +{((avgPers - avgBaseline) * 100).toFixed(1)}%
                                    </span>
                                </div>
                                <div className="w-full h-1.5 bg-gray-700 rounded-full overflow-hidden">
                                    <div className="h-full bg-green-500 w-3/4"></div>
                                </div>
                            </div>

                            <div className="flex-1 overflow-y-auto p-4 space-y-3">
                                {rounds.length > 0 && (rounds[rounds.length - 1]?.clients || []).map(c => {
                                    // Find improvement for this client
                                    const base = metrics?.before_fl?.[c.id] || 0.5;
                                    const curr = metrics?.after_personalization?.[c.id] || c.auroc;
                                    const diff = curr - base;

                                    return (
                                        <div key={c.id} className="flex items-center justify-between p-3 rounded-lg bg-black/20 border border-white/5">
                                            <div className="flex items-center gap-3">
                                                <div className={`w - 8 h - 8 rounded - lg flex items - center justify - center text - xs font - bold ${diff > 0 ? 'bg-green-500/10 text-green-400' : 'bg-red-500/10 text-red-400'
                                                    } `}>
                                                    {c.id}
                                                </div>
                                                <div>
                                                    <div className="text-xs text-gray-400">Hospital {c.id}</div>
                                                    <div className="text-sm font-bold text-white">{curr.toFixed(3)}</div>
                                                </div>
                                            </div>
                                            <div className={`text - xs font - bold ${diff > 0 ? 'text-green-500' : 'text-red-500'} `}>
                                                {diff > 0 ? '+' : ''}{(diff * 100).toFixed(1)}%
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    </div>
                </>
            )}
        </div>
    );
}

function KpiCard({ icon, label, value, sub, color, bg }) {
    return (
        <div className="admin-card p-5 rounded-xl bg-dark-card border border-white/5 hover:border-white/10 transition-colors group">
            <div className={`w - 10 h - 10 rounded - lg flex items - center justify - center mb - 4 ${bg} ${color} group - hover: scale - 110 transition - transform`}>
                {icon}
            </div>
            <div className="text-2xl font-bold text-white tracking-tight">{value}</div>
            <div className="font-medium text-gray-400 text-sm">{label}</div>
        </div>
    );
}
