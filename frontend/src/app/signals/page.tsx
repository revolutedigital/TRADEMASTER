"use client";

import { usePortfolio } from "@/hooks/usePortfolio";
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
import { timeAgo } from "@/lib/utils";
import { cn } from "@/lib/utils";

function strengthBar(strength: number) {
  const abs = Math.abs(strength);
  const widthPct = `${(abs * 100).toFixed(0)}%`;
  const color = strength > 0 ? "bg-green-500" : strength < 0 ? "bg-red-500" : "bg-gray-500";

  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-20 rounded-full bg-[var(--color-border)]">
        <div className={cn("h-full rounded-full", color)} style={{ width: widthPct }} />
      </div>
      <span className="font-mono text-xs">{strength.toFixed(3)}</span>
    </div>
  );
}

export default function SignalsPage() {
  const { signals } = usePortfolio();

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">AI Signals</h1>
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-green-400 animate-pulse-glow" />
          <span className="text-xs text-[var(--color-text-muted)]">Real-time</span>
        </div>
      </div>

      {/* Signal Legend */}
      <Card className="flex items-center gap-6 p-3">
        <span className="text-xs text-[var(--color-text-muted)]">Signal Strength:</span>
        <div className="flex items-center gap-1">
          <div className="h-3 w-3 rounded-sm bg-green-500" />
          <span className="text-xs">BUY (&ge;0.3)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="h-3 w-3 rounded-sm bg-indigo-500" />
          <span className="text-xs">HOLD (-0.3 to 0.3)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="h-3 w-3 rounded-sm bg-red-500" />
          <span className="text-xs">SELL (&le;-0.3)</span>
        </div>
      </Card>

      {/* Signals Table */}
      <Card>
        <CardHeader>
          <CardTitle>Signal History</CardTitle>
          <Badge variant="primary">{signals.length} signals</Badge>
        </CardHeader>

        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Time</TableHead>
              <TableHead>Symbol</TableHead>
              <TableHead>Action</TableHead>
              <TableHead>Strength</TableHead>
              <TableHead>Confidence</TableHead>
              <TableHead>Model</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {signals.length === 0 ? (
              <TableRow>
                <TableCell colSpan={6} className="py-12 text-center text-[var(--color-text-muted)]">
                  No signals generated yet. The ML pipeline will generate signals when market data flows.
                </TableCell>
              </TableRow>
            ) : (
              signals.map((sig) => (
                <TableRow key={sig.id}>
                  <TableCell className="text-xs text-[var(--color-text-muted)]">
                    {timeAgo(sig.created_at)}
                  </TableCell>
                  <TableCell className="font-semibold">{sig.symbol}</TableCell>
                  <TableCell>
                    <Badge
                      variant={
                        sig.action === "BUY"
                          ? "success"
                          : sig.action === "SELL"
                          ? "danger"
                          : "primary"
                      }
                    >
                      {sig.action}
                    </Badge>
                  </TableCell>
                  <TableCell>{strengthBar(sig.strength)}</TableCell>
                  <TableCell>
                    <span className="font-mono text-xs">
                      {(sig.confidence * 100).toFixed(1)}%
                    </span>
                  </TableCell>
                  <TableCell className="text-xs text-[var(--color-text-muted)]">
                    {sig.model_source}
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
