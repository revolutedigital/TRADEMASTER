"use client";

import { useState } from "react";
import { Sidebar } from "@/components/ui/sidebar";

interface OptimizationResult {
  optimal_weights: Record<string, number>;
  expected_return: number;
  expected_risk: number;
  sharpe_ratio: number;
  frontier: { risk: number; return_pct: number }[];
}

export default function OptimizerPage() {
  const [riskTolerance, setRiskTolerance] = useState(0.5);
  const [result, setResult] = useState<OptimizationResult | null>(null);
  const [loading, setLoading] = useState(false);

  async function runOptimization() {
    setLoading(true);
    try {
      const res = await fetch(`/api/v1/portfolio/optimize?risk_tolerance=${riskTolerance}`, { credentials: "include" });
      if (res.ok) setResult(await res.json());
    } catch {} finally { setLoading(false); }
  }

  return (
    <div className="flex h-screen bg-[#0a0e17]">
      <Sidebar />
      <main className="flex-1 overflow-y-auto p-6">
        <h1 className="text-2xl font-bold text-white mb-6">Portfolio Optimizer</h1>

        <div className="bg-[#141922] rounded-xl p-6 mb-6">
          <h2 className="text-sm text-gray-400 mb-4">Markowitz Mean-Variance Optimization</h2>
          <div className="flex items-center gap-6">
            <div className="flex-1">
              <label className="block text-sm text-gray-400 mb-2">Risk Tolerance: {(riskTolerance * 100).toFixed(0)}%</label>
              <input type="range" min="0" max="1" step="0.05" value={riskTolerance} onChange={(e) => setRiskTolerance(parseFloat(e.target.value))} className="w-full accent-blue-500" />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>Conservative</span>
                <span>Moderate</span>
                <span>Aggressive</span>
              </div>
            </div>
            <button onClick={runOptimization} disabled={loading} className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded-lg font-medium transition-colors">
              {loading ? "Optimizing..." : "Optimize"}
            </button>
          </div>
        </div>

        {result && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-[#141922] rounded-xl p-6">
              <h2 className="text-sm text-gray-400 mb-4">Optimal Allocation</h2>
              <div className="space-y-4">
                {Object.entries(result.optimal_weights).map(([asset, weight]) => (
                  <div key={asset}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-white font-medium">{asset}</span>
                      <span className="text-gray-300">{(weight * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <div className="bg-blue-500 h-2 rounded-full transition-all" style={{ width: `${weight * 100}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-[#141922] rounded-xl p-6">
              <h2 className="text-sm text-gray-400 mb-4">Portfolio Metrics</h2>
              <div className="space-y-4">
                <div className="flex justify-between p-3 bg-[#1a1f2e] rounded-lg">
                  <span className="text-gray-400">Expected Annual Return</span>
                  <span className="text-green-400 font-semibold">{(result.expected_return * 100).toFixed(2)}%</span>
                </div>
                <div className="flex justify-between p-3 bg-[#1a1f2e] rounded-lg">
                  <span className="text-gray-400">Expected Risk (Volatility)</span>
                  <span className="text-yellow-400 font-semibold">{(result.expected_risk * 100).toFixed(2)}%</span>
                </div>
                <div className="flex justify-between p-3 bg-[#1a1f2e] rounded-lg">
                  <span className="text-gray-400">Sharpe Ratio</span>
                  <span className="text-blue-400 font-semibold">{result.sharpe_ratio.toFixed(3)}</span>
                </div>
              </div>
            </div>

            {result.frontier.length > 0 && (
              <div className="bg-[#141922] rounded-xl p-6 col-span-1 md:col-span-2">
                <h2 className="text-sm text-gray-400 mb-4">Efficient Frontier</h2>
                <div className="h-64 flex items-end gap-1">
                  {result.frontier.map((point, i) => (
                    <div key={i} className="flex-1 flex flex-col items-center">
                      <div className="w-2 h-2 rounded-full bg-blue-500 mb-1" style={{ marginBottom: `${point.return_pct * 300}px` }} />
                    </div>
                  ))}
                </div>
                <div className="flex justify-between text-xs text-gray-500 mt-2">
                  <span>Low Risk</span>
                  <span>High Risk</span>
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
