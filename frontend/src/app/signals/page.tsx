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
        <h1 className="text-2xl font-bold">Sinais de IA</h1>
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-green-400 animate-pulse-glow" />
          <span className="text-xs text-[var(--color-text-muted)]">Tempo real</span>
        </div>
      </div>

      {/* Signal Legend */}
      <Card className="flex flex-wrap items-center gap-4 sm:gap-6 p-3">
        <span className="text-xs text-[var(--color-text-muted)]">Força do Sinal:</span>
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
          <CardTitle>Histórico de Sinais</CardTitle>
          <Badge variant="primary">{signals.length} signals</Badge>
        </CardHeader>

        <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Tempo</TableHead>
              <TableHead>Symbol</TableHead>
              <TableHead>Ação</TableHead>
              <TableHead>Força</TableHead>
              <TableHead>Confiança</TableHead>
              <TableHead>Origem</TableHead>
              <TableHead>Execução</TableHead>
              <TableHead>Evidência</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {signals.length === 0 ? (
              <TableRow>
                <TableCell colSpan={8} className="py-12 text-center text-[var(--color-text-muted)]">
                  Nenhum candidato de estratégia gerado ainda. Os sinais aprovados aparecerão com as evidências da decisão.
                </TableCell>
              </TableRow>
            ) : (
              signals.map((sig) => (
                <TableRow key={sig.id}>
                  <TableCell className="text-xs text-[var(--color-text-muted)]">
                    {timeAgo(sig.generated_at)}
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
                  <TableCell>
                    <Badge variant={sig.was_executed ? "success" : "warning"}>
                      {sig.was_executed ? "Executado" : "Não executado"}
                    </Badge>
                  </TableCell>
                  <TableCell className="min-w-64 text-xs text-[var(--color-text-muted)]">
                    {sig.evidence ? (
                      <details>
                        <summary className="cursor-pointer text-[var(--color-text)]">
                          {sig.evidence.votes.length} voto{sig.evidence.votes.length === 1 ? "" : "s"} · {sig.evidence.regime.market}
                        </summary>
                        <div className="mt-2 space-y-1">
                          <p>
                            Limite {sig.evidence.signal_threshold.toFixed(2)} · consenso {(sig.evidence.agreement_ratio * 100).toFixed(0)}% · ATR {(sig.evidence.atr_pct * 100).toFixed(2)}%
                          </p>
                          <p>
                            {sig.evidence.votes.map((vote) => `${vote.model}: ${vote.action}`).join(" · ")}
                          </p>
                        </div>
                      </details>
                    ) : (
                      "Evidência indisponível para registro legado"
                    )}
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
        </div>
      </Card>
    </div>
  );
}
