"use client";

import { useState, useEffect } from "react";
import { useMarketData } from "@/hooks/useMarketData";
import { usePortfolio } from "@/hooks/usePortfolio";
import { CandlestickChart } from "@/components/charts/candlestick-chart";
import { StatCard } from "@/components/ui/stat-card";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from "@/components/ui/table";
import { LivePrice } from "@/components/ui/live-price";
import { formatCurrency, formatPercent, timeAgo, apiFetch } from "@/lib/utils";
import { cn } from "@/lib/utils";
import type { TimeInterval } from "@/lib/types";
import {
  DollarSign,
  TrendingUp,
  Shield,
  BarChart3,
  Play,
  Square,
  Zap,
  ArrowUpCircle,
  ArrowDownCircle,
  X,
} from "lucide-react";

const intervals: TimeInterval[] = ["1m", "5m", "15m", "1h", "4h", "1d"];

export default function DashboardPage() {
  const {
    currentPrice,
    currentKlines,
    selectedSymbol,
    selectedInterval,
    setSelectedSymbol,
    setSelectedInterval,
  } = useMarketData();

  const { positions, signals, summary, riskStatus, fetchPositions, fetchSummary } = usePortfolio();

  const [engineRunning, setEngineRunning] = useState(false);
  const [engineLoading, setEngineLoading] = useState(false);
  const [orderLoading, setOrderLoading] = useState(false);
  const [lastOrder, setLastOrder] = useState<{ status: string; message: string } | null>(null);
  const [quantity, setQuantity] = useState("0.001");

  // Check engine status on mount
  useEffect(() => {
    apiFetch<{ engine_running: boolean }>("/api/v1/trading/engine/status")
      .then((s) => setEngineRunning(s.engine_running))
      .catch(() => {});
  }, []);

  const toggleEngine = async () => {
    setEngineLoading(true);
    try {
      if (engineRunning) {
        await apiFetch("/api/v1/trading/engine/stop", { method: "POST" });
        setEngineRunning(false);
      } else {
        await apiFetch("/api/v1/trading/engine/start", { method: "POST" });
        setEngineRunning(true);
      }
    } catch (err) {
      console.error("Engine toggle failed:", err);
    }
    setEngineLoading(false);
  };

  const executePaperOrder = async (side: "BUY" | "SELL") => {
    setOrderLoading(true);
    setLastOrder(null);
    try {
      const result = await apiFetch<Record<string, unknown>>("/api/v1/trading/paper-order", {
        method: "POST",
        body: JSON.stringify({
          symbol: selectedSymbol,
          side,
          quantity: parseFloat(quantity),
          stop_loss_pct: 0.02,
          take_profit_pct: 0.04,
        }),
      });
      const status = result.status as string;
      const price = result.entry_price || result.price;
      const pnl = result.realized_pnl;
      let message = "";
      if (status === "position_opened") {
        message = `${side} ${selectedSymbol} @ $${Number(price).toLocaleString()} | SL: $${Number(result.stop_loss).toLocaleString()} | TP: $${Number(result.take_profit).toLocaleString()}`;
      } else if (status === "position_closed") {
        message = `Closed opposite position | P&L: $${pnl}`;
      } else if (status === "position_increased") {
        message = `Added to ${result.side} | Avg: $${Number(result.avg_entry).toLocaleString()} | Qty: ${result.total_quantity}`;
      }
      setLastOrder({ status: status === "position_closed" ? (Number(pnl) >= 0 ? "profit" : "loss") : "ok", message });
      fetchPositions();
      fetchSummary();
    } catch (err) {
      setLastOrder({ status: "error", message: String(err) });
    }
    setOrderLoading(false);
  };

  const closePosition = async (posId: string) => {
    try {
      const result = await apiFetch<{ pnl: number }>(`/api/v1/trading/close-position/${posId}`, {
        method: "POST",
      });
      setLastOrder({
        status: result.pnl >= 0 ? "profit" : "loss",
        message: `Position closed | P&L: $${result.pnl.toFixed(2)}`,
      });
      fetchPositions();
      fetchSummary();
    } catch (err) {
      setLastOrder({ status: "error", message: String(err) });
    }
  };

  return (
    <div className="space-y-6">
      {/* Trading Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-semibold">Dashboard</h1>
          <Badge variant={engineRunning ? "success" : "default"}>
            {engineRunning ? "Engine Running" : "Engine Stopped"}
          </Badge>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant={engineRunning ? "danger" : "primary"}
            size="sm"
            onClick={toggleEngine}
            disabled={engineLoading}
          >
            {engineRunning ? (
              <>
                <Square className="mr-1.5 h-3.5 w-3.5" />
                Stop Engine
              </>
            ) : (
              <>
                <Play className="mr-1.5 h-3.5 w-3.5" />
                Start Engine
              </>
            )}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={async () => {
              try {
                const status = await apiFetch<{ engine_running: boolean }>("/api/v1/trading/engine/status");
                setEngineRunning(status.engine_running);
              } catch { /* ignore */ }
            }}
          >
            <Zap className="mr-1.5 h-3.5 w-3.5" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-4 gap-4">
        <StatCard
          label="Total Equity"
          value={formatCurrency(summary?.total_equity ?? 10000)}
          change={summary ? formatPercent(summary.daily_pnl_pct) : undefined}
          positive={summary ? summary.daily_pnl_pct >= 0 : undefined}
          icon={<DollarSign className="h-4 w-4" />}
        />
        <StatCard
          label="Daily P&L"
          value={formatCurrency(summary?.daily_pnl ?? 0)}
          change={summary ? formatPercent(summary.daily_pnl_pct) : undefined}
          positive={summary ? summary.daily_pnl >= 0 : undefined}
          icon={<TrendingUp className="h-4 w-4" />}
        />
        <StatCard
          label="Open Positions"
          value={String(summary?.open_positions ?? 0)}
          change={`${formatPercent(summary?.exposure_pct ?? 0)} exposed`}
          icon={<BarChart3 className="h-4 w-4" />}
        />
        <StatCard
          label="Risk Status"
          value={riskStatus?.circuit_breaker_state ?? "NORMAL"}
          change={`${formatPercent(riskStatus?.daily_drawdown ?? 0)} daily DD`}
          positive={riskStatus?.can_trade}
          icon={<Shield className="h-4 w-4" />}
        />
      </div>

      {/* Chart + Trading Panel */}
      <div className="grid grid-cols-[1fr_300px] gap-4">
        {/* Chart */}
        <Card>
          <CardHeader>
            <div className="flex items-center gap-3">
              {["BTCUSDT", "ETHUSDT"].map((s) => (
                <Button
                  key={s}
                  variant={selectedSymbol === s ? "primary" : "ghost"}
                  size="sm"
                  onClick={() => setSelectedSymbol(s)}
                >
                  {s.replace("USDT", "/USDT")}
                </Button>
              ))}
            </div>

            <div className="flex items-center gap-1">
              {intervals.map((iv) => (
                <Button
                  key={iv}
                  variant={selectedInterval === iv ? "primary" : "ghost"}
                  size="sm"
                  onClick={() => setSelectedInterval(iv)}
                >
                  {iv}
                </Button>
              ))}
            </div>
          </CardHeader>

          <CandlestickChart data={currentKlines} height={400} />

          {currentPrice && (
            <div className="mt-3 flex items-center justify-between border-t border-[var(--color-border)] px-4 pt-3 pb-3">
              <div className="flex items-center gap-4 text-sm">
                <span className="text-[var(--color-text-muted)]">Price</span>
                <LivePrice price={currentPrice.price} className="font-semibold text-base" />
                <span
                  className={cn(
                    "text-xs font-medium",
                    (currentPrice.change_24h ?? 0) >= 0
                      ? "text-green-400"
                      : "text-red-400"
                  )}
                >
                  {formatPercent(currentPrice.change_24h)}
                </span>
              </div>
              <div className="flex items-center gap-4 text-xs text-[var(--color-text-muted)]">
                <span>H: {formatCurrency(currentPrice.high_24h)}</span>
                <span>L: {formatCurrency(currentPrice.low_24h)}</span>
                <span>Vol: {formatCurrency(currentPrice.volume_24h, 0)}</span>
              </div>
            </div>
          )}
        </Card>

        {/* Trading Panel */}
        <Card className="flex flex-col">
          <CardHeader>
            <CardTitle>Paper Trading</CardTitle>
            <Badge variant="warning">Simulated</Badge>
          </CardHeader>

          <div className="flex flex-col gap-4 p-4 pt-0">
            {/* Symbol & Price */}
            <div className="text-center">
              <div className="text-sm text-[var(--color-text-muted)]">{selectedSymbol}</div>
              <div className="text-2xl font-bold">
                {currentPrice ? <LivePrice price={currentPrice.price} /> : "---"}
              </div>
            </div>

            {/* Quantity Input */}
            <div>
              <label className="text-xs text-[var(--color-text-muted)] mb-1 block">
                Quantity ({selectedSymbol.replace("USDT", "")})
              </label>
              <input
                type="number"
                value={quantity}
                onChange={(e) => setQuantity(e.target.value)}
                step={selectedSymbol === "BTCUSDT" ? "0.001" : "0.01"}
                min="0"
                className="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
              />
              {currentPrice && (
                <div className="text-xs text-[var(--color-text-muted)] mt-1">
                  ~ {formatCurrency(currentPrice.price * parseFloat(quantity || "0"))}
                </div>
              )}
            </div>

            {/* Quick quantity buttons */}
            <div className="flex gap-1">
              {(selectedSymbol === "BTCUSDT"
                ? ["0.001", "0.005", "0.01", "0.05"]
                : ["0.01", "0.05", "0.1", "0.5"]
              ).map((q) => (
                <Button
                  key={q}
                  variant="ghost"
                  size="sm"
                  className="flex-1 text-xs"
                  onClick={() => setQuantity(q)}
                >
                  {q}
                </Button>
              ))}
            </div>

            {/* Buy / Sell Buttons */}
            <div className="grid grid-cols-2 gap-2">
              <Button
                variant="primary"
                className="bg-green-600 hover:bg-green-700"
                onClick={() => executePaperOrder("BUY")}
                disabled={orderLoading}
              >
                <ArrowUpCircle className="mr-1.5 h-4 w-4" />
                Buy / Long
              </Button>
              <Button
                variant="danger"
                onClick={() => executePaperOrder("SELL")}
                disabled={orderLoading}
              >
                <ArrowDownCircle className="mr-1.5 h-4 w-4" />
                Sell / Short
              </Button>
            </div>

            {/* Order feedback */}
            {lastOrder && (
              <div
                className={cn(
                  "rounded-lg px-3 py-2 text-xs",
                  lastOrder.status === "ok" && "bg-indigo-500/10 text-indigo-300",
                  lastOrder.status === "profit" && "bg-green-500/10 text-green-300",
                  lastOrder.status === "loss" && "bg-red-500/10 text-red-300",
                  lastOrder.status === "error" && "bg-red-500/10 text-red-400"
                )}
              >
                {lastOrder.message}
              </div>
            )}

            {/* Risk info */}
            <div className="mt-auto border-t border-[var(--color-border)] pt-3 text-xs text-[var(--color-text-muted)] space-y-1">
              <div className="flex justify-between">
                <span>Stop Loss</span>
                <span>2.0%</span>
              </div>
              <div className="flex justify-between">
                <span>Take Profit</span>
                <span>4.0%</span>
              </div>
              <div className="flex justify-between">
                <span>Fee</span>
                <span>0.1%</span>
              </div>
            </div>
          </div>
        </Card>
      </div>

      {/* Bottom panels */}
      <div className="grid grid-cols-2 gap-4">
        {/* Open Positions */}
        <Card>
          <CardHeader>
            <CardTitle>Open Positions</CardTitle>
            <Badge variant={positions.length > 0 ? "primary" : "default"}>
              {positions.length}
            </Badge>
          </CardHeader>

          {positions.length === 0 ? (
            <p className="py-8 text-center text-sm text-[var(--color-text-muted)]">
              No open positions
            </p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Symbol</TableHead>
                  <TableHead>Side</TableHead>
                  <TableHead>Entry</TableHead>
                  <TableHead>P&L</TableHead>
                  <TableHead></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {positions.slice(0, 10).map((pos) => (
                  <TableRow key={pos.id}>
                    <TableCell className="font-medium">{pos.symbol}</TableCell>
                    <TableCell>
                      <Badge variant={pos.side === "LONG" ? "success" : "danger"}>
                        {pos.side}
                      </Badge>
                    </TableCell>
                    <TableCell className="font-mono text-xs">
                      {formatCurrency(pos.entry_price)}
                    </TableCell>
                    <TableCell
                      className={cn(
                        "font-mono text-xs",
                        pos.unrealized_pnl >= 0 ? "text-green-400" : "text-red-400"
                      )}
                    >
                      {formatCurrency(pos.unrealized_pnl)}
                    </TableCell>
                    <TableCell>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 w-6 p-0 text-red-400 hover:text-red-300"
                        onClick={() => closePosition(pos.id)}
                        title="Close position"
                      >
                        <X className="h-3.5 w-3.5" />
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </Card>

        {/* Recent Signals */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Signals</CardTitle>
            <Badge variant="primary">{signals.length}</Badge>
          </CardHeader>

          {signals.length === 0 ? (
            <p className="py-8 text-center text-sm text-[var(--color-text-muted)]">
              No signals yet
            </p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Symbol</TableHead>
                  <TableHead>Action</TableHead>
                  <TableHead>Strength</TableHead>
                  <TableHead>Time</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {signals.slice(0, 5).map((sig) => (
                  <TableRow key={sig.id}>
                    <TableCell className="font-medium">{sig.symbol}</TableCell>
                    <TableCell>
                      <Badge
                        variant={
                          sig.action === "BUY"
                            ? "success"
                            : sig.action === "SELL"
                            ? "danger"
                            : "default"
                        }
                      >
                        {sig.action}
                      </Badge>
                    </TableCell>
                    <TableCell className="font-mono text-xs">
                      {sig.strength.toFixed(3)}
                    </TableCell>
                    <TableCell className="text-xs text-[var(--color-text-muted)]">
                      {timeAgo(sig.created_at)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </Card>
      </div>
    </div>
  );
}
