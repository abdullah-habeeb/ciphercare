import { useState, useEffect, useRef } from 'react';
import { Play, RotateCcw, CheckCircle, Shield, Database, Zap, Link, RefreshCw, Terminal } from 'lucide-react';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

export default function FLSimulator() {
    const [isRunning, setIsRunning] = useState(false);
    const [logs, setLogs] = useState([]);
    const [currentRound, setCurrentRound] = useState(3);
    const [blockchainBlocks, setBlockchainBlocks] = useState(60);
    const [progress, setProgress] = useState(0);
    const [latestAuroc, setLatestAuroc] = useState({});
    const logContainerRef = useRef(null);

    // Fetch initial status
    useEffect(() => {
        fetchStatus();
    }, []);

    const fetchStatus = async () => {
        try {
            const res = await axios.get(`${API_URL}/api/status`);
            setCurrentRound(res.data.round_current);
            setBlockchainBlocks(res.data.blockchain_blocks);
        } catch (err) {
            console.error("Failed to fetch status:", err);
        }
    };

    const addLog = (type, text) => {
        setLogs(prev => [...prev, { id: Date.now(), type, text }]);
        if (logContainerRef.current) {
            setTimeout(() => {
                logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
            }, 50);
        }
    };

    const runSimulation = async () => {
        setIsRunning(true);
        setLogs([]);
        setProgress(0);

        const nextRound = currentRound + 1;

        // Step 1: Starting
        addLog('info', `[FL Server] Starting Round ${nextRound} simulation...`);
        setProgress(10);
        await sleep(500);

        // Step 2: Requesting updates
        addLog('info', `[FL Server] Requesting model updates from all 5 clients...`);
        setProgress(20);
        await sleep(500);

        // Step 3: Client updates
        const clients = [
            { id: 'A', samples: 19601 },
            { id: 'D', samples: 3000 },
            { id: 'B', samples: 1000 },
            { id: 'C', samples: 200 },
            { id: 'E', samples: 3000 },
        ];

        for (let i = 0; i < clients.length; i++) {
            await sleep(200);
            addLog('success', `[Hospital ${clients[i].id}] Sent encrypted update`);
            setProgress(20 + (i + 1) * 8);
        }

        // Step 4: DP processing
        await sleep(400);
        addLog('dp', `[DP Engine] Clipping gradients (max_norm=1.0)`);
        setProgress(65);

        await sleep(400);
        addLog('dp', `[DP Engine] Adding Gaussian noise: σ = 0.00081`);
        setProgress(70);

        // Step 5: Call REAL backend to simulate round
        addLog('info', `[Aggregator] Computing FedProxFairness weights...`);
        setProgress(75);

        try {
            const response = await axios.post(`${API_URL}/api/simulate_round`);
            const data = response.data;

            setLatestAuroc(data.auroc);

            await sleep(300);
            const weightStr = Object.entries(data.weights)
                .map(([k, v]) => `${k}:${(v * 100).toFixed(0)}%`)
                .join(' | ');
            addLog('info', `[Aggregator] Weights: ${weightStr}`);
            setProgress(80);

            await sleep(300);
            addLog('success', `[Aggregator] Global model updated successfully!`);
            setProgress(85);

            // Step 6: Blockchain
            await sleep(300);
            addLog('blockchain', `[Blockchain] Creating new block...`);
            setProgress(90);

            await sleep(400);
            addLog('blockchain', `[Blockchain] Block ${data.blockchain_block} committed ✅`);
            setProgress(95);

            // Step 7: Complete
            await sleep(300);
            const avgAuroc = Object.values(data.auroc).reduce((a, b) => a + b, 0) / 5;
            addLog('success', `[FL Server] Round ${data.round} complete! Global AUROC: ${avgAuroc.toFixed(3)}`);
            setProgress(100);

            // Update state
            setCurrentRound(data.round);
            setBlockchainBlocks(data.blockchain_block + 1);

        } catch (err) {
            console.error("Simulation error:", err);
            addLog('error', `[ERROR] Backend simulation failed: ${err.message}`);
        }

        setIsRunning(false);
    };

    const resetSimulation = () => {
        setLogs([]);
        setProgress(0);
        setLatestAuroc({});
        fetchStatus();
    };

    return (
        <div className="p-8 space-y-8 h-full overflow-y-auto">
            {/* Header */}
            <div className="flex justify-between items-end">
                <div>
                    <h1 className="text-3xl font-bold text-white tracking-tight">Live FL Controller</h1>
                    <p className="text-gray-400 mt-1 text-sm">Real-time Federation Orchestrator</p>
                </div>
                <div className="flex gap-3">
                    <button
                        onClick={fetchStatus}
                        className="flex items-center gap-2 px-4 py-2 bg-dark-card border border-gray-700 text-gray-300 rounded-lg hover:border-gray-500 hover:text-white transition-all text-sm"
                    >
                        <RefreshCw size={16} /> Status
                    </button>

                    <button
                        onClick={resetSimulation}
                        disabled={isRunning}
                        className="flex items-center gap-2 px-4 py-2 bg-dark-card border border-gray-700 text-gray-300 rounded-lg hover:border-gray-500 hover:text-white transition-all text-sm disabled:opacity-50"
                    >
                        <RotateCcw size={16} /> Reset
                    </button>

                    <button
                        onClick={runSimulation}
                        disabled={isRunning}
                        className="flex items-center gap-2 px-6 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-bold text-sm shadow-lg shadow-blue-500/20 disabled:opacity-50 transition-all"
                    >
                        <Play size={16} /> {isRunning ? 'Running Protocol...' : 'Execute Round'}
                    </button>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Controls & Status */}
                <div className="space-y-6">
                    {/* Progress Panel */}
                    <div className="admin-card rounded-2xl p-6 bg-dark-card">
                        <div className="flex justify-between p-1 mb-4">
                            <span className="text-sm font-bold text-gray-400 uppercase tracking-wider">Round Progress</span>
                            <span className="text-sm font-mono text-blue-400">{Math.round(progress)}%</span>
                        </div>
                        <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                            <div
                                className="h-full bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 transition-all duration-300 ease-out shadow-[0_0_10px_rgba(59,130,246,0.5)]"
                                style={{ width: `${progress}%` }}
                            />
                        </div>
                        <div className="mt-8 grid grid-cols-2 gap-4">
                            <StatusItem label="Current Round " value={currentRound} icon={<Database size={16} />} />
                            <StatusItem label="Ledger Height" value={blockchainBlocks} icon={<Link size={16} />} />
                            <StatusItem label="Active Clients" value="5 / 5" icon={<Zap size={16} />} />
                            <StatusItem label="Privacy Cost" value="ε = 5.0" icon={<Shield size={16} />} />
                        </div>
                    </div>

                    {/* Real-time Results */}
                    {Object.keys(latestAuroc).length > 0 && (
                        <div className="admin-card rounded-2xl p-6 border-l-4 border-l-green-500 animate-in fade-in slide-in-from-right-4">
                            <h4 className="text-sm font-bold text-green-400 mb-4 flex items-center gap-2">
                                <CheckCircle size={16} /> Round Complete
                            </h4>
                            <div className="space-y-3">
                                {Object.entries(latestAuroc).map(([h, auroc]) => (
                                    <div key={h} className="flex justify-between items-center text-sm">
                                        <span className="text-gray-400">Hospital {h}</span>
                                        <div className="flex items-center gap-3">
                                            <div className="w-24 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                                                <div className="h-full bg-green-500" style={{ width: `${auroc * 100}%` }}></div>
                                            </div>
                                            <span className="font-mono text-white font-bold">{auroc.toFixed(3)}</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>

                {/* Terminal */}
                <div className="lg:col-span-2 admin-card rounded-2xl bg-[#0f1115] border border-gray-800 flex flex-col overflow-hidden h-[500px] shadow-2xl">
                    <div className="flex items-center justify-between px-4 py-3 bg-[#161920] border-b border-gray-800">
                        <div className="flex items-center gap-2">
                            <Terminal size={14} className="text-gray-500" />
                            <span className="text-gray-400 text-xs font-mono">fl_process_daemon --verbose</span>
                        </div>
                        <div className="flex gap-2">
                            <div className="w-2.5 h-2.5 rounded-full bg-red-500/20 border border-red-500/50"></div>
                            <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/20 border border-yellow-500/50"></div>
                            <div className="w-2.5 h-2.5 rounded-full bg-green-500/20 border border-green-500/50"></div>
                        </div>
                    </div>

                    <div
                        ref={logContainerRef}
                        className="flex-1 p-4 overflow-y-auto font-mono text-xs space-y-2 custom-scrollbar"
                    >
                        {logs.length === 0 && (
                            <div className="text-gray-600 text-center mt-20">
                                <div className="mb-2 opacity-50">System Idle. Ready to initialize protocol.</div>
                                <div className="text-[10px] opacity-30">Waiting for trigger command...</div>
                            </div>
                        )}
                        {logs.map((log) => (
                            <LogEntry key={log.id} type={log.type} text={log.text} />
                        ))}
                        {isRunning && (
                            <div className="w-2 h-4 bg-blue-500/50 animate-pulse ml-1"></div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function StatusItem({ label, value, icon }) {
    return (
        <div className="p-3 bg-black/20 rounded-lg border border-white/5 flex items-center justify-between">
            <div className="flex items-center gap-2 text-gray-500 text-xs font-bold uppercase">
                {icon} {label}
            </div>
            <div className="text-white font-mono font-bold">{value}</div>
        </div>
    )
}

function LogEntry({ type, text }) {
    const colors = {
        info: 'text-blue-300',
        success: 'text-green-300',
        dp: 'text-orange-300',
        blockchain: 'text-purple-300',
        error: 'text-red-400',
    };

    const prefixes = {
        info: 'ℹ',
        success: '✔',
        dp: '🔒',
        blockchain: '⛓',
        error: '✖',
    };

    return (
        <div className={`flex items-start gap-3 ${colors[type]} border-l-2 border-current pl-3 py-0.5 bg-white/5 rounded-r`}>
            <span className="opacity-70 mt-0.5 text-[10px]">{prefixes[type]}</span>
            <span className="leading-relaxed">{text}</span>
        </div>
    );
}
