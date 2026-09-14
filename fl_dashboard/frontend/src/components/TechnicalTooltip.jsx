import React, { useState } from 'react';
import { HelpCircle, Info } from 'lucide-react';
import { GLOSSARY } from '../utils/glossary';

export default function TechnicalTooltip({ term, definition, children, highlight = true }) {
    const [isVisible, setIsVisible] = useState(false);

    const text = definition || GLOSSARY[term] || "No definition available.";
    const displayTerm = children || term;

    return (
        <span
            className="relative inline-block cursor-help group"
            onMouseEnter={() => setIsVisible(true)}
            onMouseLeave={() => setIsVisible(false)}
        >
            <span className={`inline-flex items-center gap-1 ${highlight ? 'underline decoration-dotted decoration-gray-500 hover:decoration-blue-400 hover:text-blue-300' : ''} transition-colors`}>
                {displayTerm}
                {highlight && <Info size={10} className="text-gray-500 opacity-50 group-hover:opacity-100 group-hover:text-blue-400" />}
            </span>

            {isVisible && (
                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 z-50 animate-in fade-in zoom-in-95 duration-150">
                    <div className="relative bg-gray-900/95 backdrop-blur-xl border border-gray-700 text-left p-3 rounded-xl shadow-2xl shadow-black/50">
                        {/* Arrow */}
                        <div className="absolute -bottom-1.5 left-1/2 -translate-x-1/2 w-3 h-3 bg-gray-900 border-r border-b border-gray-700 rotate-45"></div>

                        <div className="relative text-xs">
                            <div className="font-bold text-blue-300 mb-1 flex items-center gap-2">
                                <HelpCircle size={12} />
                                {term}
                            </div>
                            <div className="text-gray-300 leading-relaxed">
                                {text}
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </span>
    );
}
