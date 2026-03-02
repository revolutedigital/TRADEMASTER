"use client";

import { useState, useEffect } from "react";
import { Sidebar } from "@/components/ui/sidebar";

interface BacktestResult {
  id: string;
  strategy_name: string;
  symbol: string;
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  total_trades: number;
  created_at: string;
}

export default function BacktestComparePage() {
  const [backtests, setBacktests] = useState<BacktestResult[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchBacktests();
  }, []);

  async function fetchBacktests() {
    try {
      const res = await fetch("/api/v1/backtest/history", { credentials: "include" });
      if (res.ok) setBacktests(await res.json());
    } catch {} finally { setLoading(false); }
  }

  function toggleSelect(id: string) {
    setSelected((prev) => prev.includes(id) ? prev.filter((s) => s !== id) : prev.length < 4 ? [...prev, id] : prev);
  }

  const selectedResults = backtests.filter((b) => selected.includes(b.id));
  const metrics = ["total_return", "sharpe_ratio", "max_drawdown", "win_rate", "total_trades"] as const;
  const metricLabels: Record<string, string> = {
    total_return: "Total Return", sharpe_ratio: "Sharpe Ratio", max_drawdown: "Max Drawdown", win_rate: "Win Rate", total_trades: "Total Trades",
  };

  function formatMetric(key: string, value: number) {
    switch (key) {
      case "total_return": case "max_drawdown": case "win_rate": return `${(value * 100).toFixed(2)}%`;
      case "sharpe_ratio": return value.toFixed(3);
      default: return value.toString();
    }
  }

  function getBestValue(key: string, values: number[]) {
    if (key === "max_drawdown") return Math.max(...values);
    return Math.max(...values);
  }

  return (
    <div className="flex h-screen bg-[#0a0e17]">
      <Sidebar />
      <main className="flex-1 overflow-y-auto p-6">
        <h1 className="text-2xl font-bold text-white mb-6">Compare Backtests</h1>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1">
            <div className="bg-[#141922] rounded-xl p-4">
              <h2 className="text-sm text-gray-400 mb-3">Select up to 4 backtests</h2>
              {loading ? (
                <div className="space-y-2">{[...Array(5)].map((_, i) => <div key={i} className="bg-[#1a1f2e] rounded-lg p-3 animate-pulse h-16" />)}</div>
              ) : (
                <div className="space-y-2 max-h-[60vh] overflow-y-auto">
                  {backtests.map((bt) => (
                    <button key={bt.id} onClick={() => toggleSelect(bt.id)} className={`w-full text-left p-3 rounded-lg transition-colors ${selected.includes(bt.id) ? "bg-blue-600/20 border border-blue-500" : "bg-[#1a1f2e] hover:bg-[#252a38] border border-transparent"}`}>
                      <div className="text-white text-sm font-medium">{bt.strategy_name}</div>
                      <div className="text-xs text-gray-400">{bt.symbol} - {new Date(bt.created_at).toLocaleDateString()}</div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="lg:col-span-2">
            {selectedResults.length === 0 ? (
              <div className="bg-[#141922] rounded-xl p-12 text-center">
                <p className="text-gray-400">Select backtests from the left panel to compare</p>
              </div>
            ) : (
              <div className="bg-[#141922] rounded-xl p-6">
                <h2 className="text-sm text-gray-400 mb-4">Comparison Table</h2>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-gray-700">
                        <th className="text-left text-sm text-gray-400 pb-3 pr-4">Metric</th>
                        {selectedResults.map((bt) => (
                          <th key={bt.id} className="text-center text-sm text-gray-400 pb-3 px-2">
                            <div className="text-white">{bt.strategy_name}</div>
                            <div className="text-xs">{bt.symbol}</div>
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {metrics.map((metric) => {
                        const values = selectedResults.map((bt) => bt[metric]);
                        const best = getBestValue(metric, values);
                        return (
                          <tr key={metric} className="border-b border-gray-800">
                            <td className="text-sm text-gray-300 py-3 pr-4">{metricLabels[metric]}</td>
                            {selectedResults.map((bt) => (
                              <td key={bt.id} className={`text-center py-3 px-2 text-sm ${bt[metric] === best ? "text-green-400 font-semibold" : "text-white"}`}>
                                {formatMetric(metric, bt[metric])}
                              </td>
                            ))}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
