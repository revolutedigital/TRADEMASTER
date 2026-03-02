"use client";

import { useState } from "react";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { StatCard } from "@/components/ui/stat-card";
import { apiFetch, formatCurrency, formatPercent } from "@/lib/utils";
import type { BacktestResult } from "@/lib/types";
import { Sliders, Play, Loader2, BarChart3 } from "lucide-react";

interface IndicatorConfig {
  id: string;
  label: string;
  enabled: boolean;
  params: { label: string; key: string; value: number; min: number; max: number; step: number }[];
}

const defaultIndicators: IndicatorConfig[] = [
  {
    id: "sma",
    label: "SMA (Simple Moving Average)",
    enabled: false,
    params: [
      { label: "Short Period", key: "sma_short", value: 10, min: 2, max: 100, step: 1 },
      { label: "Long Period", key: "sma_long", value: 30, min: 5, max: 200, step: 1 },
    ],
  },
  {
    id: "ema",
    label: "EMA (Exponential Moving Average)",
    enabled: false,
    params: [
      { label: "Short Period", key: "ema_short", value: 12, min: 2, max: 100, step: 1 },
      { label: "Long Period", key: "ema_long", value: 26, min: 5, max: 200, step: 1 },
    ],
  },
  {
    id: "rsi",
    label: "RSI (Relative Strength Index)",
    enabled: false,
    params: [
      { label: "Period", key: "rsi_period", value: 14, min: 2, max: 50, step: 1 },
      { label: "Overbought", key: "rsi_overbought", value: 70, min: 50, max: 95, step: 1 },
      { label: "Oversold", key: "rsi_oversold", value: 30, min: 5, max: 50, step: 1 },
    ],
  },
  {
    id: "macd",
    label: "MACD",
    enabled: false,
    params: [
      { label: "Fast Period", key: "macd_fast", value: 12, min: 2, max: 50, step: 1 },
      { label: "Slow Period", key: "macd_slow", value: 26, min: 5, max: 100, step: 1 },
      { label: "Signal Period", key: "macd_signal", value: 9, min: 2, max: 50, step: 1 },
    ],
  },
  {
    id: "bollinger",
    label: "Bollinger Bands",
    enabled: false,
    params: [
      { label: "Period", key: "bb_period", value: 20, min: 5, max: 100, step: 1 },
      { label: "Std Deviation", key: "bb_std", value: 2, min: 0.5, max: 4, step: 0.1 },
    ],
  },
];

export default function StrategyBuilderPage() {
  const [indicators, setIndicators] = useState<IndicatorConfig[]>(defaultIndicators);
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [interval, setInterval] = useState("1h");
  const [capital, setCapital] = useState(10000);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<BacktestResult | null>(null);

  const enabledIndicators = indicators.filter((ind) => ind.enabled);

  const toggleIndicator = (id: string) => {
    setIndicators((prev) =>
      prev.map((ind) =>
        ind.id === id ? { ...ind, enabled: !ind.enabled } : ind
      )
    );
  };

  const updateParam = (indicatorId: string, paramKey: string, value: number) => {
    setIndicators((prev) =>
      prev.map((ind) =>
        ind.id === indicatorId
          ? {
              ...ind,
              params: ind.params.map((p) =>
                p.key === paramKey ? { ...p, value } : p
              ),
            }
          : ind
      )
    );
  };

  const runBacktest = async () => {
    if (enabledIndicators.length === 0) {
      setError("Please select at least one indicator.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Build indicator params payload
      const indicatorParams: Record<string, unknown> = {};
      for (const ind of enabledIndicators) {
        indicatorParams[ind.id] = {};
        for (const p of ind.params) {
          (indicatorParams[ind.id] as Record<string, number>)[p.key] = p.value;
        }
      }

      const data = await apiFetch<BacktestResult>("/api/v1/backtest/run", {
        method: "POST",
        body: JSON.stringify({
          symbol,
          interval,
          initial_capital: capital,
          indicators: Object.keys(indicatorParams),
          indicator_params: indicatorParams,
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
      {/* Header */}
      <div className="flex items-center gap-3">
        <Sliders className="h-6 w-6 text-[var(--color-primary)]" />
        <h1 className="text-2xl font-bold">Strategy Builder</h1>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column - Configuration */}
        <div className="space-y-6">
          {/* General Settings */}
          <Card>
            <CardHeader>
              <CardTitle>General Settings</CardTitle>
            </CardHeader>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div>
                <label className="mb-1 block text-xs text-[var(--color-text-muted)]">
                  Symbol
                </label>
                <select
                  value={symbol}
                  onChange={(e) => setSymbol(e.target.value)}
                  className="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm text-[var(--color-text)]"
                >
                  <option value="BTCUSDT">BTC/USDT</option>
                  <option value="ETHUSDT">ETH/USDT</option>
                </select>
              </div>

              <div>
                <label className="mb-1 block text-xs text-[var(--color-text-muted)]">
                  Interval
                </label>
                <select
                  value={interval}
                  onChange={(e) => setInterval(e.target.value)}
                  className="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm text-[var(--color-text)]"
                >
                  <option value="15m">15 minutes</option>
                  <option value="1h">1 hour</option>
                  <option value="4h">4 hours</option>
                  <option value="1d">1 day</option>
                </select>
              </div>

              <div>
                <label className="mb-1 block text-xs text-[var(--color-text-muted)]">
                  Initial Capital ($)
                </label>
                <input
                  type="number"
                  value={capital}
                  onChange={(e) => setCapital(Number(e.target.value))}
                  className="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm text-[var(--color-text)]"
                />
              </div>
            </div>
          </Card>

          {/* Indicators */}
          <Card>
            <CardHeader>
              <CardTitle>Indicators</CardTitle>
              <Badge variant="primary">
                {enabledIndicators.length} selected
              </Badge>
            </CardHeader>

            <div className="space-y-4">
              {indicators.map((ind) => (
                <div
                  key={ind.id}
                  className="rounded-lg border border-[var(--color-border)] p-3"
                >
                  <label className="flex items-center gap-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={ind.enabled}
                      onChange={() => toggleIndicator(ind.id)}
                      className="h-4 w-4 rounded border-[var(--color-border)] accent-[var(--color-primary)]"
                    />
                    <span className="text-sm font-medium">{ind.label}</span>
                  </label>

                  {ind.enabled && (
                    <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-3 pl-7">
                      {ind.params.map((param) => (
                        <div key={param.key}>
                          <label className="mb-1 block text-xs text-[var(--color-text-muted)]">
                            {param.label}
                          </label>
                          <input
                            type="number"
                            value={param.value}
                            min={param.min}
                            max={param.max}
                            step={param.step}
                            onChange={(e) =>
                              updateParam(ind.id, param.key, Number(e.target.value))
                            }
                            className="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-1.5 text-sm text-[var(--color-text)]"
                          />
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* Right Column - Preview & Results */}
        <div className="space-y-6">
          {/* Preview */}
          <Card>
            <CardHeader>
              <CardTitle>Strategy Preview</CardTitle>
            </CardHeader>

            <div className="space-y-3">
              <div className="flex items-center justify-between text-sm">
                <span className="text-[var(--color-text-muted)]">Symbol</span>
                <span className="font-medium">
                  {symbol.replace("USDT", "/USDT")}
                </span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-[var(--color-text-muted)]">Interval</span>
                <span className="font-medium">{interval}</span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-[var(--color-text-muted)]">Capital</span>
                <span className="font-medium font-mono">
                  {formatCurrency(capital)}
                </span>
              </div>

              <div className="border-t border-[var(--color-border)] pt-3">
                <p className="mb-2 text-xs text-[var(--color-text-muted)]">
                  Selected Indicators
                </p>
                {enabledIndicators.length === 0 ? (
                  <p className="text-sm text-[var(--color-text-muted)]">
                    No indicators selected. Choose at least one to run a
                    backtest.
                  </p>
                ) : (
                  <div className="flex flex-wrap gap-2">
                    {enabledIndicators.map((ind) => (
                      <Badge key={ind.id} variant="primary">
                        {ind.id.toUpperCase()}
                      </Badge>
                    ))}
                  </div>
                )}
              </div>

              <div className="flex items-center justify-between text-sm border-t border-[var(--color-border)] pt-3">
                <span className="text-[var(--color-text-muted)]">
                  Signal Count (est.)
                </span>
                <span className="font-mono font-medium">
                  {enabledIndicators.length > 0
                    ? enabledIndicators.length * 2
                    : 0}{" "}
                  signals
                </span>
              </div>
            </div>

            <div className="mt-4">
              <Button
                variant="primary"
                size="lg"
                className="w-full"
                onClick={runBacktest}
                disabled={loading || enabledIndicators.length === 0}
              >
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Running Backtest...
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
            <Card>
              <CardHeader>
                <CardTitle>
                  <div className="flex items-center gap-2">
                    <BarChart3 className="h-4 w-4" />
                    Backtest Results
                  </div>
                </CardTitle>
                <Badge variant={result.total_return >= 0 ? "success" : "danger"}>
                  {result.total_trades} trades
                </Badge>
              </CardHeader>

              <div className="grid grid-cols-2 gap-3">
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
                <StatCard
                  label="Total Trades"
                  value={String(result.total_trades)}
                  positive={result.total_trades > 0}
                />
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
