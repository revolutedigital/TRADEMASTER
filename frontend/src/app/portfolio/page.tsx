"use client";

import { usePortfolio } from "@/hooks/usePortfolio";
import { StatCard } from "@/components/ui/stat-card";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from "@/components/ui/table";
import { formatCurrency, formatPercent, timeAgo } from "@/lib/utils";
import { cn } from "@/lib/utils";
import {
  DollarSign,
  TrendingUp,
  TrendingDown,
  PieChart,
  Shield,
  AlertTriangle,
} from "lucide-react";

export default function PortfolioPage() {
  const { positions, summary, riskStatus } = usePortfolio();

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Portfolio</h1>

      {/* Stats */}
      <div className="grid grid-cols-5 gap-4">
        <StatCard
          label="Total Equity"
          value={formatCurrency(summary?.total_equity ?? 10000)}
          icon={<DollarSign className="h-4 w-4" />}
        />
        <StatCard
          label="Available Balance"
          value={formatCurrency(summary?.available_balance ?? 10000)}
          icon={<DollarSign className="h-4 w-4" />}
        />
        <StatCard
          label="Unrealized P&L"
          value={formatCurrency(summary?.total_unrealized_pnl ?? 0)}
          positive={summary ? summary.total_unrealized_pnl >= 0 : undefined}
          icon={<TrendingUp className="h-4 w-4" />}
        />
        <StatCard
          label="Realized P&L"
          value={formatCurrency(summary?.total_realized_pnl ?? 0)}
          positive={summary ? summary.total_realized_pnl >= 0 : undefined}
          icon={<TrendingDown className="h-4 w-4" />}
        />
        <StatCard
          label="Exposure"
          value={formatPercent(summary?.exposure_pct ?? 0)}
          change={`${summary?.open_positions ?? 0} positions`}
          icon={<PieChart className="h-4 w-4" />}
        />
      </div>

      {/* Risk Status */}
      <Card>
        <CardHeader>
          <CardTitle>Risk Management</CardTitle>
          <Badge
            variant={
              riskStatus?.circuit_breaker_state === "NORMAL"
                ? "success"
                : riskStatus?.circuit_breaker_state === "HALTED"
                ? "danger"
                : "warning"
            }
          >
            {riskStatus?.circuit_breaker_state ?? "NORMAL"}
          </Badge>
        </CardHeader>

        <div className="grid grid-cols-4 gap-4">
          <div className="flex items-center gap-3 rounded-lg bg-[var(--color-background)] p-3">
            <Shield className="h-5 w-5 text-[var(--color-text-muted)]" />
            <div>
              <p className="text-xs text-[var(--color-text-muted)]">Can Trade</p>
              <p className={cn("text-sm font-medium", riskStatus?.can_trade ? "text-green-400" : "text-red-400")}>
                {riskStatus?.can_trade ? "Yes" : "No"}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3 rounded-lg bg-[var(--color-background)] p-3">
            <AlertTriangle className="h-5 w-5 text-[var(--color-text-muted)]" />
            <div>
              <p className="text-xs text-[var(--color-text-muted)]">Daily Drawdown</p>
              <p className={cn("text-sm font-mono font-medium",
                (riskStatus?.daily_drawdown ?? 0) > 0.02 ? "text-red-400" : "text-[var(--color-text)]"
              )}>
                {formatPercent(riskStatus?.daily_drawdown ?? 0)}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3 rounded-lg bg-[var(--color-background)] p-3">
            <AlertTriangle className="h-5 w-5 text-[var(--color-text-muted)]" />
            <div>
              <p className="text-xs text-[var(--color-text-muted)]">Weekly Drawdown</p>
              <p className="text-sm font-mono font-medium">
                {formatPercent(riskStatus?.weekly_drawdown ?? 0)}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3 rounded-lg bg-[var(--color-background)] p-3">
            <TrendingDown className="h-5 w-5 text-[var(--color-text-muted)]" />
            <div>
              <p className="text-xs text-[var(--color-text-muted)]">Position Size Mult.</p>
              <p className="text-sm font-mono font-medium">
                {riskStatus?.position_size_multiplier?.toFixed(1) ?? "1.0"}x
              </p>
            </div>
          </div>
        </div>
      </Card>

      {/* Positions Table */}
      <Card>
        <CardHeader>
          <CardTitle>Open Positions</CardTitle>
          <Badge variant="primary">{positions.length}</Badge>
        </CardHeader>

        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Symbol</TableHead>
              <TableHead>Side</TableHead>
              <TableHead>Quantity</TableHead>
              <TableHead>Entry Price</TableHead>
              <TableHead>Current Price</TableHead>
              <TableHead>Unrealized P&L</TableHead>
              <TableHead>Stop Loss</TableHead>
              <TableHead>Take Profit</TableHead>
              <TableHead>Opened</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {positions.length === 0 ? (
              <TableRow>
                <TableCell colSpan={9} className="py-12 text-center text-[var(--color-text-muted)]">
                  No open positions. The AI will open trades when strong signals are detected.
                </TableCell>
              </TableRow>
            ) : (
              positions.map((pos) => (
                <TableRow key={pos.id}>
                  <TableCell className="font-semibold">{pos.symbol}</TableCell>
                  <TableCell>
                    <Badge variant={pos.side === "LONG" ? "success" : "danger"}>
                      {pos.side}
                    </Badge>
                  </TableCell>
                  <TableCell className="font-mono text-xs">{pos.quantity.toFixed(6)}</TableCell>
                  <TableCell className="font-mono text-xs">{formatCurrency(pos.entry_price)}</TableCell>
                  <TableCell className="font-mono text-xs">{formatCurrency(pos.current_price)}</TableCell>
                  <TableCell className={cn(
                    "font-mono text-xs font-medium",
                    pos.unrealized_pnl >= 0 ? "text-green-400" : "text-red-400"
                  )}>
                    {formatCurrency(pos.unrealized_pnl)}
                  </TableCell>
                  <TableCell className="font-mono text-xs text-red-400">
                    {pos.stop_loss_price ? formatCurrency(pos.stop_loss_price) : "-"}
                  </TableCell>
                  <TableCell className="font-mono text-xs text-green-400">
                    {pos.take_profit_price ? formatCurrency(pos.take_profit_price) : "-"}
                  </TableCell>
                  <TableCell className="text-xs text-[var(--color-text-muted)]">
                    {timeAgo(pos.opened_at)}
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </Card>
    </div>
  );
}
