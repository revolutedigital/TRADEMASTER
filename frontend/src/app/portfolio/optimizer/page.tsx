"use client";

import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { PageHeader } from "@/components/ui/page-header";
import { Spinner } from "@/components/ui/progress";
import { Progress } from "@/components/ui/progress";

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
    <div className="space-y-6">
      <PageHeader title="Portfolio Optimizer" description="Markowitz Mean-Variance Optimization for optimal asset allocation" />

      <Card>
        <CardContent>
          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-6">
            <div className="flex-1 w-full">
              <label className="block text-xs font-medium uppercase tracking-wider text-[var(--color-text-muted)] mb-2">
                Risk Tolerance: <span className="tabular-nums text-[var(--color-text)]">{(riskTolerance * 100).toFixed(0)}%</span>
              </label>
              <input
                type="range" min="0" max="1" step="0.05"
                value={riskTolerance}
                onChange={(e) => setRiskTolerance(parseFloat(e.target.value))}
                className="w-full accent-[var(--color-primary)]"
              />
              <div className="mt-1 flex justify-between text-xs text-[var(--color-text-muted)]">
                <span>Conservative</span>
                <span>Moderate</span>
                <span>Aggressive</span>
              </div>
            </div>
            <Button variant="primary" onClick={runOptimization} disabled={loading} className="min-w-[140px]">
              {loading ? <Spinner size="sm" /> : "Optimize"}
            </Button>
          </div>
        </CardContent>
      </Card>

      {result && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-fade-in">
          <Card>
            <CardContent>
              <h2 className="text-xs font-medium uppercase tracking-wider text-[var(--color-text-muted)] mb-4">Optimal Allocation</h2>
              <div className="space-y-4">
                {Object.entries(result.optimal_weights).map(([asset, weight]) => (
                  <div key={asset}>
                    <div className="mb-1 flex justify-between text-sm">
                      <span className="font-medium text-[var(--color-text)]">{asset}</span>
                      <span className="tabular-nums text-[var(--color-text-muted)]">{(weight * 100).toFixed(1)}%</span>
                    </div>
                    <Progress value={weight * 100} variant="gradient" size="sm" />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent>
              <h2 className="text-xs font-medium uppercase tracking-wider text-[var(--color-text-muted)] mb-4">Portfolio Metrics</h2>
              <div className="space-y-3">
                <div className="flex justify-between rounded-[var(--radius-md)] bg-[var(--color-background)] p-3">
                  <span className="text-sm text-[var(--color-text-muted)]">Expected Annual Return</span>
                  <span className="font-semibold tabular-nums text-[var(--color-success)]">{(result.expected_return * 100).toFixed(2)}%</span>
                </div>
                <div className="flex justify-between rounded-[var(--radius-md)] bg-[var(--color-background)] p-3">
                  <span className="text-sm text-[var(--color-text-muted)]">Expected Risk (Volatility)</span>
                  <span className="font-semibold tabular-nums text-[var(--color-warning)]">{(result.expected_risk * 100).toFixed(2)}%</span>
                </div>
                <div className="flex justify-between rounded-[var(--radius-md)] bg-[var(--color-background)] p-3">
                  <span className="text-sm text-[var(--color-text-muted)]">Sharpe Ratio</span>
                  <span className="font-semibold tabular-nums text-[var(--color-primary)]">{result.sharpe_ratio.toFixed(3)}</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {result.frontier.length > 0 && (
            <Card className="md:col-span-2">
              <CardContent>
                <h2 className="text-xs font-medium uppercase tracking-wider text-[var(--color-text-muted)] mb-4">Efficient Frontier</h2>
                <div className="flex h-64 items-end gap-1">
                  {result.frontier.map((point, i) => (
                    <div key={i} className="flex flex-1 flex-col items-center">
                      <div
                        className="h-2 w-2 rounded-full bg-[var(--color-primary)] transition-all"
                        style={{ marginBottom: `${point.return_pct * 300}px` }}
                      />
                    </div>
                  ))}
                </div>
                <div className="mt-2 flex justify-between text-xs text-[var(--color-text-muted)]">
                  <span>Low Risk</span>
                  <span>High Risk</span>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}
    </div>
  );
}
