"use client";

import { useMarketData } from "@/hooks/useMarketData";
import { usePortfolio } from "@/hooks/usePortfolio";
import { CandlestickChart } from "@/components/charts/candlestick-chart";
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
import { formatCurrency, timeAgo } from "@/lib/utils";
import { cn } from "@/lib/utils";
import type { TimeInterval } from "@/lib/types";

const intervals: TimeInterval[] = ["1m", "5m", "15m", "1h", "4h", "1d"];

export default function TradingPage() {
  const {
    currentPrice,
    currentKlines,
    selectedSymbol,
    selectedInterval,
    setSelectedSymbol,
    setSelectedInterval,
  } = useMarketData();

  const { positions, orders, riskStatus } = usePortfolio();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Trading Terminal</h1>
        <div className="flex items-center gap-2">
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
          <span className="text-xs text-[var(--color-text-muted)]">
            {riskStatus?.can_trade ? "AI Trading Active" : "Trading Paused"}
          </span>
        </div>
      </div>

      {/* Main chart area */}
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

            {currentPrice && (
              <span className="ml-4 font-mono text-lg font-bold">
                {formatCurrency(currentPrice.price)}
              </span>
            )}
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

        <CandlestickChart data={currentKlines} height={500} />
      </Card>

      {/* Bottom panels */}
      <div className="grid grid-cols-2 gap-4">
        {/* Active Positions */}
        <Card>
          <CardHeader>
            <CardTitle>Active Positions</CardTitle>
            <Badge variant="primary">{positions.length}</Badge>
          </CardHeader>

          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Symbol</TableHead>
                <TableHead>Side</TableHead>
                <TableHead>Qty</TableHead>
                <TableHead>Entry</TableHead>
                <TableHead>Current</TableHead>
                <TableHead>P&L</TableHead>
                <TableHead>SL</TableHead>
                <TableHead>TP</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {positions.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={8} className="text-center text-[var(--color-text-muted)]">
                    No active positions
                  </TableCell>
                </TableRow>
              ) : (
                positions.map((pos) => (
                  <TableRow key={pos.id}>
                    <TableCell className="font-medium">{pos.symbol}</TableCell>
                    <TableCell>
                      <Badge variant={pos.side === "LONG" ? "success" : "danger"}>
                        {pos.side}
                      </Badge>
                    </TableCell>
                    <TableCell className="font-mono text-xs">{pos.quantity.toFixed(6)}</TableCell>
                    <TableCell className="font-mono text-xs">
                      {formatCurrency(pos.entry_price)}
                    </TableCell>
                    <TableCell className="font-mono text-xs">
                      {formatCurrency(pos.current_price)}
                    </TableCell>
                    <TableCell
                      className={cn(
                        "font-mono text-xs font-medium",
                        pos.unrealized_pnl >= 0 ? "text-green-400" : "text-red-400"
                      )}
                    >
                      {formatCurrency(pos.unrealized_pnl)}
                    </TableCell>
                    <TableCell className="font-mono text-xs text-red-400">
                      {pos.stop_loss_price ? formatCurrency(pos.stop_loss_price) : "-"}
                    </TableCell>
                    <TableCell className="font-mono text-xs text-green-400">
                      {pos.take_profit_price ? formatCurrency(pos.take_profit_price) : "-"}
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </Card>

        {/* Recent Orders */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Orders</CardTitle>
            <Badge>{orders.length}</Badge>
          </CardHeader>

          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Symbol</TableHead>
                <TableHead>Side</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Price</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Time</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {orders.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={6} className="text-center text-[var(--color-text-muted)]">
                    No orders yet
                  </TableCell>
                </TableRow>
              ) : (
                orders.slice(0, 10).map((order) => (
                  <TableRow key={order.id}>
                    <TableCell className="font-medium">{order.symbol}</TableCell>
                    <TableCell>
                      <Badge variant={order.side === "BUY" ? "success" : "danger"}>
                        {order.side}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-xs">{order.type}</TableCell>
                    <TableCell className="font-mono text-xs">
                      {formatCurrency(order.price)}
                    </TableCell>
                    <TableCell>
                      <Badge
                        variant={
                          order.status === "FILLED"
                            ? "success"
                            : order.status === "CANCELLED"
                            ? "danger"
                            : "warning"
                        }
                      >
                        {order.status}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-xs text-[var(--color-text-muted)]">
                      {timeAgo(order.created_at)}
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </Card>
      </div>
    </div>
  );
}
