import { FileText, Download, Code, Terminal } from 'lucide-react';

export default function TechnicalDeepDive() {
    return (
        <div className="p-8 space-y-8 max-w-5xl mx-auto">
            <div className="text-center mb-12">
                <h1 className="text-3xl font-bold text-gray-900">Technical Deep Dive</h1>
                <p className="text-gray-500 mt-2">Architecture, Configurations, and Implementation Details</p>
            </div>

            {/* Downloads Section */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
                <DownloadCard
                    title="Full Technical Report"
                    desc="Markdown report with extensive metrics (COMPREHENSIVE_TECHNICAL_REPORT.md)"
                    icon={<FileText />}
                />
                <DownloadCard
                    title="Raw Simulation Logs"
                    desc="Complete JSON logs for all 3 FL rounds and metrics"
                    icon={<Terminal />}
                />
            </div>

            {/* Architecture */}
            <div className="space-y-8">
                <Section title="Fairness Aggregation Formula">
                    <CodeBlock>
                        {`def aggregate_fit_metrics(metrics: List[Tuple[int, Dict]]) -> Dict:
    """FedProxFairness: Weight aggregation logic"""
    # 1. Performance component (60%)
    w_perf = 0.6 * (auroc ** 2)
    
    # 2. Data Volume component (30%)
    w_data = 0.3 * (n_samples / total_samples)
    
    # 3. Domain Relevance component (10%)
    w_domain = 0.1 * calculate_domain_similarity(client_domain, global_domain)
    
    final_weight = w_perf + w_data + w_domain
    return normalize(final_weight)`}
                    </CodeBlock>
                </Section>

                <Section title="Differential Privacy (Gaussian Mechanism)">
                    <CodeBlock>
                        {`class PrivacyEngine:
    def __init__(self, epsilon=5.0, delta=1e-5, max_grad_norm=1.0):
        self.noise_multiplier = get_noise_multiplier(epsilon, delta)
        self.max_grad_norm = max_grad_norm

    def add_noise(self, gradients):
        # Clip gradients
        total_norm = torch.norm(gradients)
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        clipped_grads = gradients * torch.clamp(clip_coef, max=1.0)
        
        # Add Gaussian noise
        noise = torch.randn_like(gradients) * self.noise_multiplier
        return clipped_grads + noise`}
                    </CodeBlock>
                </Section>

                <Section title="Blockchain Audit Logger">
                    <CodeBlock>
                        {`def log_round_to_chain(round_data):
    """Create immutable audit record"""
    prev_hash = blockchain[-1].hash
    
    block = {
        "index": len(blockchain),
        "timestamp": datetime.utcnow().isoformat(),
        "type": "FL_ROUND",
        "data": round_data,
        "prev_hash": prev_hash
    }
    
    # Proof of Integrity
    block["hash"] = sha256(json.dumps(block, sort_keys=True).encode()).hexdigest()
    blockchain.append(block)
    save_chain()`}
                    </CodeBlock>
                </Section>
            </div>
        </div>
    );
}

function DownloadCard({ title, desc, icon }) {
    return (
        <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm hover:border-blue-300 hover:shadow-md transition-all cursor-pointer group">
            <div className="flex items-start gap-4">
                <div className="p-3 bg-blue-50 text-blue-600 rounded-lg group-hover:bg-blue-600 group-hover:text-white transition-colors">
                    {icon}
                </div>
                <div>
                    <h3 className="font-semibold text-gray-900">{title}</h3>
                    <p className="text-sm text-gray-500 mt-1 mb-3">{desc}</p>
                    <div className="flex items-center gap-1 text-sm font-medium text-blue-600">
                        <Download size={16} /> Download
                    </div>
                </div>
            </div>
        </div>
    )
}

function Section({ title, children }) {
    return (
        <div className="border border-gray-200 rounded-xl overflow-hidden">
            <div className="bg-gray-50 px-6 py-3 font-semibold text-gray-700 border-b border-gray-200 flex items-center gap-2">
                <Code size={18} /> {title}
            </div>
            <div className="bg-slate-900 p-6 overflow-x-auto">
                {children}
            </div>
        </div>
    )
}

function CodeBlock({ children }) {
    return (
        <pre className="font-mono text-sm text-blue-300">
            {children}
        </pre>
    )
}
