"use client";

import { useState, useEffect } from "react";
import { GitCompare } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { PageHeader } from "@/components/ui/page-header";
import { EmptyState } from "@/components/ui/empty-state";
import { Spinner } from "@/components/ui/progress";

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
    <div className="space-y-6">
      <PageHeader title="Compare Backtests" description="Select up to 4 backtests to compare side-by-side" />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <Card>
            <CardContent>
              <h2 className="text-xs font-medium uppercase tracking-wider text-[var(--color-text-muted)] mb-3">
                Select Backtests ({selected.length}/4)
              </h2>
              {loading ? (
                <div className="flex justify-center py-8"><Spinner /></div>
              ) : backtests.length === 0 ? (
                <p className="py-4 text-center text-sm text-[var(--color-text-muted)]">No backtests found. Run a backtest first.</p>
              ) : (
                <div className="space-y-2 max-h-[60vh] overflow-y-auto pr-1">
                  {backtests.map((bt) => (
                    <button
                      key={bt.id}
                      onClick={() => toggleSelect(bt.id)}
                      className={`w-full text-left rounded-[var(--radius-md)] p-3 transition-all ${
                        selected.includes(bt.id)
                          ? "bg-[var(--color-primary)]/10 border border-[var(--color-primary)] shadow-sm"
                          : "bg-[var(--color-background)] border border-transparent hover:border-[var(--color-border)]"
                      }`}
                    >
                      <div className="text-sm font-medium text-[var(--color-text)]">{bt.strategy_name}</div>
                      <div className="text-xs text-[var(--color-text-muted)]">
                        {bt.symbol} &middot; {new Date(bt.created_at).toLocaleDateString()}
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        <div className="lg:col-span-2">
          {selectedResults.length === 0 ? (
            <Card>
              <EmptyState
                icon={<GitCompare className="h-7 w-7" />}
                title="No backtests selected"
                description="Select backtests from the left panel to compare their performance metrics."
              />
            </Card>
          ) : (
            <Card>
              <CardContent>
                <h2 className="text-xs font-medium uppercase tracking-wider text-[var(--color-text-muted)] mb-4">Comparison Table</h2>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-[var(--color-border)]">
                        <th className="pb-3 pr-4 text-left text-xs font-medium uppercase tracking-wider text-[var(--color-text-muted)]">Metric</th>
                        {selectedResults.map((bt) => (
                          <th key={bt.id} className="px-2 pb-3 text-center">
                            <div className="text-sm font-medium text-[var(--color-text)]">{bt.strategy_name}</div>
                            <div className="text-xs text-[var(--color-text-muted)]">{bt.symbol}</div>
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {metrics.map((metric) => {
                        const values = selectedResults.map((bt) => bt[metric]);
                        const best = getBestValue(metric, values);
                        return (
                          <tr key={metric} className="border-b border-[var(--color-border)]/50">
                            <td className="py-3 pr-4 text-sm text-[var(--color-text-muted)]">{metricLabels[metric]}</td>
                            {selectedResults.map((bt) => (
                              <td
                                key={bt.id}
                                className={`px-2 py-3 text-center text-sm tabular-nums ${
                                  bt[metric] === best ? "font-semibold text-[var(--color-success)]" : "text-[var(--color-text)]"
                                }`}
                              >
                                {formatMetric(metric, bt[metric])}
                              </td>
                            ))}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
