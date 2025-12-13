import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useState, useEffect } from 'react';
import axios from 'axios';
import { RefreshCw, Download } from 'lucide-react';

const API_URL = 'http://localhost:8000';

export default function TrainingProgress() {
    const [selectedRound, setSelectedRound] = useState(null);
    const [roundData, setRoundData] = useState([]);
    const [allRounds, setAllRounds] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchRounds();
    }, []);

    const fetchRounds = async () => {
        setLoading(true);
        try {
            const res = await axios.get(`${API_URL}/api/rounds`);
            const rounds = res.data || [];
            setAllRounds(rounds);

            const chartData = rounds.map(r => {
                const entry = { round: `Rd ${r.round}` };
                (r.clients || []).forEach(c => {
                    entry[c.id] = c.auroc;
                });
                return entry;
            });
            setRoundData(chartData);

            if (rounds.length > 0) {
                setSelectedRound(rounds[rounds.length - 1]);
            }
        } catch (err) {
            console.error("Failed to fetch rounds:", err);
        }
        setLoading(false);
    };

    const handleRoundSelect = (roundNum) => {
        const round = allRounds.find(r => r.round === roundNum);
        if (round) setSelectedRound(round);
    };

    return (
        <div className="p-8 space-y-8 h-full overflow-y-auto">
            <div className="flex justify-between items-end">
                <div>
                    <h1 className="text-3xl font-bold text-white tracking-tight">Global Convergence</h1>
                    <p className="text-gray-400 mt-1 text-sm">Real-time AUROC progression across participating nodes</p>
                </div>

                <div className="flex items-center gap-4">
                    <div className="flex items-center bg-dark-card border border-gray-700 rounded-lg p-1">
                        <span className="text-xs font-bold text-gray-500 uppercase px-3">Round</span>
                        <select
                            value={selectedRound?.round || ''}
                            onChange={(e) => handleRoundSelect(Number(e.target.value))}
                            className="bg-transparent text-white text-sm font-bold focus:outline-none py-1 pr-2"
                        >
                            {[...allRounds].reverse().map(r => (
                                <option key={r.round} value={r.round}>#{r.round}</option>
                            ))}
                        </select>
                    </div>
                    <button onClick={fetchRounds} className="p-2 bg-dark-card text-gray-400 hover:text-white rounded-lg border border-gray-700 hover:border-gray-500 transition-all">
                        <RefreshCw size={18} />
                    </button>
                </div>
            </div>

            {loading ? (
                <div className="h-96 flex items-center justify-center text-gray-500">Loading metrics...</div>
            ) : roundData.length === 0 ? (
                <div className="h-96 flex flex-col items-center justify-center text-gray-500 border border-dashed border-gray-700 rounded-xl">
                    <Download size={48} className="mb-4 opacity-20" />
                    <p>No training convergence data available</p>
                </div>
            ) : (
                <>
                    {/* Main Chart */}
                    <div className="admin-card rounded-2xl p-6 bg-dark-card">
                        <div className="h-[400px]">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={roundData}>
                                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#374151" />
                                    <XAxis dataKey="round" tick={{ fill: '#9CA3AF' }} axisLine={false} tickLine={false} dy={10} />
                                    <YAxis domain={['auto', 'auto']} tick={{ fill: '#9CA3AF' }} axisLine={false} tickLine={false} dx={-10} tickFormatter={v => v.toFixed(2)} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#1f2937', borderRadius: '8px', border: '1px solid #374151', color: '#fff' }}
                                    />
                                    <Legend wrapperStyle={{ paddingTop: '20px' }} />
                                    <Line type="monotone" dataKey="A" stroke="#3B82F6" strokeWidth={3} dot={{ r: 4, fill: '#3B82F6' }} activeDot={{ r: 6 }} name="Hosp A (ECG)" />
                                    <Line type="monotone" dataKey="B" stroke="#DB2777" strokeWidth={3} dot={{ r: 4, fill: '#DB2777' }} name="Hosp B (Vitals)" />
                                    <Line type="monotone" dataKey="C" stroke="#F97316" strokeWidth={3} dot={{ r: 4, fill: '#F97316' }} name="Hosp C (X-Ray)" />
                                    <Line type="monotone" dataKey="D" stroke="#EF4444" strokeWidth={3} dot={{ r: 4, fill: '#EF4444' }} name="Hosp D (Geriatric)" />
                                    <Line type="monotone" dataKey="E" stroke="#10B981" strokeWidth={3} dot={{ r: 4, fill: '#10B981' }} name="Hosp E (Multi)" />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* AUROC Summary Cards */}
                    {selectedRound && (
                        <div className="grid grid-cols-5 gap-4">
                            {(selectedRound.clients || []).map(c => (
                                <div key={c.id} className="admin-card p-4 rounded-xl text-center bg-dark-card/50 hover:bg-dark-card transition-colors">
                                    <div className="text-xs font-bold text-gray-500 uppercase mb-2">Hospital {c.id}</div>
                                    <div className="text-2xl font-bold text-white mb-1">{c.auroc?.toFixed(4)}</div>
                                    <div className="flex justify-center items-center gap-2 text-[10px] text-gray-400">
                                        <span>{(c.samples || 0).toLocaleString()} samples</span>
                                        <span className="text-blue-400 font-bold">W:{(c.normalized_weight * 100)?.toFixed(1)}%</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Logs View */}
                    {selectedRound && (
                        <div className="admin-card rounded-xl p-0 overflow-hidden border border-gray-800">
                            <div className="bg-black/30 px-4 py-2 border-b border-gray-800 flex justify-between items-center">
                                <span className="text-xs font-mono text-green-400">source: fl_results/round_{selectedRound.round}_aggregation.json</span>
                                <span className="text-[10px] font-bold text-gray-500 uppercase">Raw JSON Payload</span>
                            </div>
                            <pre className="p-4 overflow-x-auto text-xs text-blue-300 font-mono bg-[#0d1117]">
                                {JSON.stringify(selectedRound, null, 2)}
                            </pre>
                        </div>
                    )}
                </>
            )}
        </div>
    );
}
