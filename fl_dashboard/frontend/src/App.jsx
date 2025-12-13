import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { LayoutDashboard, Activity, Database, ShieldCheck, Link as LinkIcon, Siren, Code, Play, Menu } from 'lucide-react';

import ExecutiveDashboard from './pages/ExecutiveDashboard';
import TrainingProgress from './pages/TrainingProgress';
import HospitalProfiles from './pages/HospitalProfiles';
import FairnessPrivacy from './pages/FairnessPrivacy';
import BlockchainExplorer from './pages/BlockchainExplorer';
import EmergencyDemo from './pages/EmergencyDemo';
import TechnicalDeepDive from './pages/TechnicalDeepDive';
import FLSimulator from './pages/FLSimulator';

function App() {
  return (
    <Router>
      <div className="flex h-screen bg-dark-bg text-gray-100 font-sans overflow-hidden">
        {/* Sidebar */}
        <aside className="w-64 bg-dark-sidebar border-r border-dark-border flex flex-col shadow-xl z-20">
          <div className="p-6 border-b border-dark-border flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-500/30">
              <ShieldCheck className="text-white" size={20} />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white tracking-tight">Cipher-Care</h1>
              <p className="text-[10px] text-gray-400 uppercase tracking-widest font-semibold">Federated Health AI</p>
            </div>
          </div>

          <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
            <div className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2 px-3 mt-2">Analytics</div>
            <NavLink to="/" icon={<LayoutDashboard size={20} />} label="Executive Dashboard" />
            <NavLink to="/training" icon={<Activity size={20} />} label="Training Progress" />
            <NavLink to="/hospitals" icon={<Database size={20} />} label="Hospital Profiles" />

            <div className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2 px-3 mt-6">Security</div>
            <NavLink to="/fairness" icon={<ShieldCheck size={20} />} label="Fairness & Privacy" />
            <NavLink to="/blockchain" icon={<LinkIcon size={20} />} label="Blockchain Audit" />

            <div className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2 px-3 mt-6">Live Modules</div>
            <NavLink to="/simulator" icon={<Play size={20} />} label="FL Simulator" highlight="blue" />
            <NavLink to="/emergency" icon={<Siren size={20} />} label="IoMT Streams" highlight="red" />

            <div className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2 px-3 mt-6">Docs</div>
            <NavLink to="/technical" icon={<Code size={20} />} label="Technical Deep Dive" />
          </nav>

          <div className="p-4 border-t border-dark-border bg-black/10">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center text-xs font-bold">AZ</div>
              <div>
                <div className="text-sm font-medium text-white">Administrator</div>
                <div className="text-xs text-green-400 flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse"></span> Online
                </div>
              </div>
            </div>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 overflow-auto bg-dark-bg relative">
          <div className="absolute inset-0 bg-gradient-to-br from-blue-900/5 via-transparent to-purple-900/5 pointer-events-none"></div>
          <Routes>
            <Route path="/" element={<ExecutiveDashboard />} />
            <Route path="/training" element={<TrainingProgress />} />
            <Route path="/hospitals" element={<HospitalProfiles />} />
            <Route path="/fairness" element={<FairnessPrivacy />} />
            <Route path="/blockchain" element={<BlockchainExplorer />} />
            <Route path="/simulator" element={<FLSimulator />} />
            <Route path="/emergency" element={<EmergencyDemo />} />
            <Route path="/technical" element={<TechnicalDeepDive />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

function NavLink({ to, icon, label, highlight }) {
  const location = useLocation();
  const isActive = location.pathname === to;

  const highlightClasses = {
    blue: isActive ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/25' : 'text-blue-400 hover:bg-blue-500/10',
    red: isActive ? 'bg-red-600 text-white shadow-lg shadow-red-500/25' : 'text-red-400 hover:bg-red-500/10',
    default: isActive ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-500/25' : 'text-gray-400 hover:bg-white/5 hover:text-white',
  };

  const styleClass = highlight ? highlightClasses[highlight] : highlightClasses.default;

  return (
    <Link
      to={to}
      className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 ${styleClass}`}
    >
      {icon}
      <span>{label}</span>
      {isActive && <div className="ml-auto w-1.5 h-1.5 rounded-full bg-white/50"></div>}
    </Link>
  );
}

export default App;
