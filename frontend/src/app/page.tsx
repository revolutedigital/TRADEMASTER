"use client";

import { useState, useEffect } from "react";
import { useMarketData } from "@/hooks/useMarketData";
import { usePortfolio } from "@/hooks/usePortfolio";
import { CandlestickChart } from "@/components/charts/candlestick-chart";
import { StatCard } from "@/components/ui/stat-card";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { EmptyState } from "@/components/ui/empty-state";
import { Tooltip } from "@/components/ui/tooltip";
import { useToast } from "@/components/ui/toast";
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from "@/components/ui/table";
import { LivePrice } from "@/components/ui/live-price";
import { OnboardingWizard } from "@/components/onboarding/wizard";
import { useOnboardingStore } from "@/stores/onboardingStore";
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
  Inbox,
  Radio,
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
  const onboardingCompleted = useOnboardingStore((s) => s.completed);
  const toast = useToast();

  const [engineRunning, setEngineRunning] = useState(false);
  const [engineLoading, setEngineLoading] = useState(false);
  const [orderLoading, setOrderLoading] = useState(false);
  const [lastOrder, setLastOrder] = useState<{ status: string; message: string } | null>(null);
  const [quantity, setQuantity] = useState("0.001");

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
        toast.warning("Trading engine stopped");
      } else {
        await apiFetch("/api/v1/trading/engine/start", { method: "POST" });
        setEngineRunning(true);
        toast.success("Trading engine started");
      }
    } catch (err) {
      toast.error(`Engine toggle failed: ${err}`);
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
        message = `${side} ${selectedSymbol} @ $${Number(price).toLocaleString()}`;
        toast.success(message);
      } else if (status === "position_closed") {
        message = `Closed position | P&L: $${pnl}`;
        Number(pnl) >= 0 ? toast.success(message) : toast.warning(message);
      } else if (status === "position_increased") {
        message = `Added to ${result.side} | Avg: $${Number(result.avg_entry).toLocaleString()}`;
        toast.info(message);
      }
      setLastOrder({ status: status === "position_closed" ? (Number(pnl) >= 0 ? "profit" : "loss") : "ok", message });
      fetchPositions();
      fetchSummary();
    } catch (err) {
      const message = String(err);
      setLastOrder({ status: "error", message });
      toast.error(message);
    }
    setOrderLoading(false);
  };

  const closePosition = async (posId: string) => {
    try {
      const result = await apiFetch<{ pnl: number }>(`/api/v1/trading/close-position/${posId}`, {
        method: "POST",
      });
      const message = `Position closed | P&L: $${result.pnl.toFixed(2)}`;
      setLastOrder({
        status: result.pnl >= 0 ? "profit" : "loss",
        message,
      });
      result.pnl >= 0 ? toast.success(message) : toast.warning(message);
      fetchPositions();
      fetchSummary();
    } catch (err) {
      toast.error(`Close failed: ${err}`);
      setLastOrder({ status: "error", message: String(err) });
    }
  };

  return (
    <div className="space-y-6">
      {!onboardingCompleted && <OnboardingWizard />}

      {/* Page Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-semibold tracking-tight">Dashboard</h1>
          <Badge variant={engineRunning ? "success" : "default"}>
            <span className={cn("mr-1.5 h-1.5 w-1.5 rounded-full inline-block", engineRunning ? "bg-[var(--color-success)] animate-pulse-glow" : "bg-[var(--color-text-faint)]")} />
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
            {engineRunning ? <Square className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
            {engineRunning ? "Stop Engine" : "Start Engine"}
          </Button>
          <Tooltip content="Refresh engine status">
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
              <Zap className="h-3.5 w-3.5" />
            </Button>
          </Tooltip>
        </div>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 stagger-children">
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
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-4">
        {/* Chart */}
        <Card>
          <CardHeader>
            <div className="flex items-center gap-1.5">
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

            <div className="flex items-center gap-0.5">
              {intervals.map((iv) => (
                <Button
                  key={iv}
                  variant={selectedInterval === iv ? "primary" : "ghost"}
                  size="sm"
                  onClick={() => setSelectedInterval(iv)}
                  className="min-w-[2rem]"
                >
                  {iv}
                </Button>
              ))}
            </div>
          </CardHeader>

          <CandlestickChart data={currentKlines} height={400} />

          {currentPrice && (
            <div className="mt-3 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 border-t border-[var(--color-border)] px-4 pt-3 pb-3">
              <div className="flex items-center gap-4 text-sm">
                <span className="text-[var(--color-text-faint)] text-xs uppercase tracking-wider">Price</span>
                <LivePrice price={currentPrice.price} className="font-semibold text-base" />
                <span
                  className={cn(
                    "text-xs font-medium tabular-nums",
                    (currentPrice.change_24h ?? 0) >= 0
                      ? "text-[var(--color-success)]"
                      : "text-[var(--color-danger)]"
                  )}
                >
                  {formatPercent(currentPrice.change_24h)}
                </span>
              </div>
              <div className="flex items-center gap-4 text-xs text-[var(--color-text-faint)] tabular-nums">
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

          <div className="flex flex-col gap-4 px-0 pt-0 flex-1">
            {/* Symbol & Price */}
            <div className="text-center py-2">
              <div className="text-xs text-[var(--color-text-faint)] uppercase tracking-wider">{selectedSymbol}</div>
              <div className="text-2xl font-bold mt-1 tabular-nums">
                {currentPrice ? <LivePrice price={currentPrice.price} /> : "---"}
              </div>
            </div>

            {/* Quantity Input */}
            <Input
              label={`Quantity (${selectedSymbol.replace("USDT", "")})`}
              type="number"
              value={quantity}
              onChange={(e) => setQuantity(e.target.value)}
              step={selectedSymbol === "BTCUSDT" ? "0.001" : "0.01"}
              min="0"
              helperText={currentPrice ? `~ ${formatCurrency(currentPrice.price * parseFloat(quantity || "0"))}` : undefined}
            />

            {/* Quick quantity buttons */}
            <div className="flex gap-1">
              {(selectedSymbol === "BTCUSDT"
                ? ["0.001", "0.005", "0.01", "0.05"]
                : ["0.01", "0.05", "0.1", "0.5"]
              ).map((q) => (
                <Button
                  key={q}
                  variant={quantity === q ? "primary" : "ghost"}
                  size="sm"
                  className="flex-1 text-xs font-mono"
                  onClick={() => setQuantity(q)}
                >
                  {q}
                </Button>
              ))}
            </div>

            {/* Buy / Sell Buttons */}
            <div className="grid grid-cols-2 gap-2">
              <Button
                variant="success"
                onClick={() => executePaperOrder("BUY")}
                disabled={orderLoading}
              >
                <ArrowUpCircle className="h-4 w-4" />
                Buy / Long
              </Button>
              <Button
                variant="danger"
                onClick={() => executePaperOrder("SELL")}
                disabled={orderLoading}
              >
                <ArrowDownCircle className="h-4 w-4" />
                Sell / Short
              </Button>
            </div>

            {/* Order feedback */}
            {lastOrder && (
              <div
                className={cn(
                  "rounded-lg px-3 py-2 text-xs animate-fade-in",
                  lastOrder.status === "ok" && "bg-[var(--color-primary-light)] text-[var(--color-primary)]",
                  lastOrder.status === "profit" && "bg-[var(--color-success-light)] text-[var(--color-success)]",
                  lastOrder.status === "loss" && "bg-[var(--color-danger-light)] text-[var(--color-danger)]",
                  lastOrder.status === "error" && "bg-[var(--color-danger-light)] text-[var(--color-danger)]"
                )}
              >
                {lastOrder.message}
              </div>
            )}

            {/* Risk info */}
            <div className="mt-auto border-t border-[var(--color-border)] pt-3 text-xs text-[var(--color-text-faint)] space-y-1.5">
              <div className="flex justify-between">
                <span>Stop Loss</span>
                <span className="font-mono">2.0%</span>
              </div>
              <div className="flex justify-between">
                <span>Take Profit</span>
                <span className="font-mono">4.0%</span>
              </div>
              <div className="flex justify-between">
                <span>Fee</span>
                <span className="font-mono">0.1%</span>
              </div>
            </div>
          </div>
        </Card>
      </div>

      {/* Bottom panels */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Open Positions */}
        <Card>
          <CardHeader>
            <CardTitle>Open Positions</CardTitle>
            <Badge variant={positions.length > 0 ? "primary" : "default"}>
              {positions.length}
            </Badge>
          </CardHeader>

          {positions.length === 0 ? (
            <EmptyState
              icon={<Inbox className="h-6 w-6" />}
              title="No open positions"
              description="Execute a paper trade to see your positions here"
            />
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Symbol</TableHead>
                  <TableHead>Side</TableHead>
                  <TableHead>Entry</TableHead>
                  <TableHead>P&L</TableHead>
                  <TableHead><span className="sr-only">Actions</span></TableHead>
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
                    <TableCell className="font-mono text-xs tabular-nums">
                      {formatCurrency(pos.entry_price)}
                    </TableCell>
                    <TableCell
                      className={cn(
                        "font-mono text-xs tabular-nums",
                        pos.unrealized_pnl >= 0 ? "text-[var(--color-success)]" : "text-[var(--color-danger)]"
                      )}
                    >
                      {formatCurrency(pos.unrealized_pnl)}
                    </TableCell>
                    <TableCell>
                      <Tooltip content="Close position">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-7 w-7 p-0 text-[var(--color-danger)] hover:bg-[var(--color-danger-light)]"
                          onClick={() => closePosition(pos.id)}
                          aria-label={`Close ${pos.symbol} position`}
                        >
                          <X className="h-3.5 w-3.5" />
                        </Button>
                      </Tooltip>
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
            <EmptyState
              icon={<Radio className="h-6 w-6" />}
              title="No signals yet"
              description="Start the trading engine to receive AI signals"
            />
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
                {signals.slice(0, 5).map((sig, i) => (
                  <TableRow key={sig.id} className={i === 0 ? "animate-fade-in" : ""}>
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
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 w-16 rounded-full bg-[var(--color-background)] overflow-hidden">
                          <div
                            className={cn(
                              "h-full rounded-full transition-all duration-500",
                              sig.strength > 0.7 ? "bg-[var(--color-success)]" :
                              sig.strength > 0.4 ? "bg-[var(--color-warning)]" :
                              "bg-[var(--color-danger)]"
                            )}
                            style={{ width: `${sig.strength * 100}%` }}
                          />
                        </div>
                        <span className="font-mono text-xs tabular-nums">
                          {sig.strength.toFixed(3)}
                        </span>
                      </div>
                    </TableCell>
                    <TableCell className="text-xs text-[var(--color-text-faint)]">
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
