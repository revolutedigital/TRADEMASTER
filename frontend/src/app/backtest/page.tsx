"use client";

import { useState } from "react";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { StatCard } from "@/components/ui/stat-card";
import { EquityChart } from "@/components/charts/equity-chart";
import { formatCurrency, formatPercent, apiFetch } from "@/lib/utils";
import type { BacktestResult } from "@/lib/types";
import { FlaskConical, Play, Loader2 } from "lucide-react";

export default function BacktestPage() {
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [interval, setInterval] = useState("1h");
  const [capital, setCapital] = useState(10000);
  const [threshold, setThreshold] = useState(0.3);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runBacktest = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiFetch<BacktestResult>("/api/v1/backtest/run", {
        method: "POST",
        body: JSON.stringify({
          symbol,
          interval,
          initial_capital: capital,
          signal_threshold: threshold,
        }),
      });
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Backtest failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <FlaskConical className="h-6 w-6 text-[var(--color-primary)]" />
        <h1 className="text-2xl font-bold">Backtesting</h1>
      </div>

      {/* Configuration */}
      <Card>
        <CardHeader>
          <CardTitle>Backtest Configuration</CardTitle>
        </CardHeader>

        <div className="grid grid-cols-4 gap-4">
          {/* Symbol */}
          <div>
            <label className="mb-1 block text-xs text-[var(--color-text-muted)]">Symbol</label>
            <select
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              className="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm"
            >
              <option value="BTCUSDT">BTC/USDT</option>
              <option value="ETHUSDT">ETH/USDT</option>
            </select>
          </div>

          {/* Interval */}
          <div>
            <label className="mb-1 block text-xs text-[var(--color-text-muted)]">Interval</label>
            <select
              value={interval}
              onChange={(e) => setInterval(e.target.value)}
              className="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm"
            >
              <option value="15m">15 minutes</option>
              <option value="1h">1 hour</option>
              <option value="4h">4 hours</option>
              <option value="1d">1 day</option>
            </select>
          </div>

          {/* Capital */}
          <div>
            <label className="mb-1 block text-xs text-[var(--color-text-muted)]">Initial Capital ($)</label>
            <input
              type="number"
              value={capital}
              onChange={(e) => setCapital(Number(e.target.value))}
              className="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm"
            />
          </div>

          {/* Threshold */}
          <div>
            <label className="mb-1 block text-xs text-[var(--color-text-muted)]">Signal Threshold</label>
            <input
              type="number"
              step="0.05"
              min="0.1"
              max="0.9"
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              className="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm"
            />
          </div>
        </div>

        <div className="mt-4 flex justify-end">
          <Button variant="primary" onClick={runBacktest} disabled={loading}>
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Running...
              </>
            ) : (
              <>
                <Play className="mr-2 h-4 w-4" />
                Run Backtest
              </>
            )}
          </Button>
        </div>
      </Card>

      {/* Error */}
      {error && (
        <Card className="border-red-500/50 bg-red-500/10">
          <p className="text-sm text-red-400">{error}</p>
        </Card>
      )}

      {/* Results */}
      {result && (
        <>
          {/* Metrics */}
          <div className="grid grid-cols-5 gap-4">
            <StatCard
              label="Total Return"
              value={formatCurrency(result.total_return)}
              change={formatPercent(result.total_return_pct)}
              positive={result.total_return >= 0}
            />
            <StatCard
              label="Win Rate"
              value={formatPercent(result.win_rate)}
              change={`${result.winning_trades}W / ${result.losing_trades}L`}
              positive={result.win_rate > 0.5}
            />
            <StatCard
              label="Profit Factor"
              value={result.profit_factor.toFixed(2)}
              positive={result.profit_factor > 1}
            />
            <StatCard
              label="Sharpe Ratio"
              value={result.sharpe_ratio.toFixed(2)}
              positive={result.sharpe_ratio > 1}
            />
            <StatCard
              label="Max Drawdown"
              value={formatPercent(result.max_drawdown)}
              positive={false}
            />
          </div>

          {/* Equity Curve */}
          <Card>
            <CardHeader>
              <CardTitle>Equity Curve</CardTitle>
              <Badge variant={result.total_return >= 0 ? "success" : "danger"}>
                {result.total_trades} trades
              </Badge>
            </CardHeader>
            <EquityChart
              data={result.equity_curve}
              height={300}
              color={result.total_return >= 0 ? "#22c55e" : "#ef4444"}
            />
          </Card>
        </>
      )}
    </div>
  );
}
