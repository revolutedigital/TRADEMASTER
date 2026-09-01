"use client";

import { useState } from "react";
import { usePortfolio } from "@/hooks/usePortfolio";
import { StatCard } from "@/components/ui/stat-card";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogFooter } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { useToast } from "@/components/ui/toast";
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from "@/components/ui/table";
import { apiFetch, formatCurrency, formatPercent, timeAgo } from "@/lib/utils";
import { cn } from "@/lib/utils";
import { ExportButton } from "@/components/ui/export-button";
import type { Position } from "@/lib/types";
import {
  DollarSign,
  TrendingUp,
  TrendingDown,
  PieChart,
  Shield,
  AlertTriangle,
} from "lucide-react";

export default function PortfolioPage() {
  const { positions, summary, riskStatus, fetchPositions, fetchSummary } = usePortfolio();
  const toast = useToast();
  const [positionToClose, setPositionToClose] = useState<Position | null>(null);
  const [closeTotpCode, setCloseTotpCode] = useState("");
  const [closeConfirmation, setCloseConfirmation] = useState("");
  const [closeLoading, setCloseLoading] = useState(false);
  const [closeError, setCloseError] = useState<string | null>(null);

  const resetCloseDialog = () => {
    if (closeLoading) return;
    setPositionToClose(null);
    setCloseTotpCode("");
    setCloseConfirmation("");
    setCloseError(null);
  };

  const closePositionOnExchange = async () => {
    if (!positionToClose) return;
    setCloseLoading(true);
    setCloseError(null);
    try {
      const result = await apiFetch<{ status: string; exit_price: number }>(
        `/api/v1/trading/positions/${positionToClose.id}/close-exchange`,
        {
          method: "POST",
          body: JSON.stringify({
            confirmation_phrase: closeConfirmation,
            totp_code: closeTotpCode,
          }),
        }
      );
      await Promise.all([fetchPositions(), fetchSummary()]);
      toast.success(
        `${positionToClose.symbol} encerrada na Binance a ${formatCurrency(result.exit_price)}`
      );
      setPositionToClose(null);
      setCloseTotpCode("");
      setCloseConfirmation("");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Não foi possível encerrar a posição na Binance";
      setCloseError(message);
      toast.error(message);
    } finally {
      setCloseLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold">Portfólio</h1>
          {summary && (
            <Badge variant={summary.execution_mode === "LIVE" ? "danger" : "warning"}>
              Livro ativo: {summary.execution_mode}
            </Badge>
          )}
        </div>
        <ExportButton endpoint="/api/v1/export/portfolio" filename="portfolio.csv" />
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
        <StatCard
          label="Patrimônio Total"
          value={formatCurrency(summary?.total_equity ?? 10000)}
          icon={<DollarSign className="h-4 w-4" />}
        />
        <StatCard
          label="Saldo Disponível"
          value={formatCurrency(summary?.available_balance ?? 10000)}
          icon={<DollarSign className="h-4 w-4" />}
        />
        <StatCard
          label="P&L Não Realizado"
          value={formatCurrency(summary?.total_unrealized_pnl ?? 0)}
          positive={summary ? summary.total_unrealized_pnl >= 0 : undefined}
          icon={<TrendingUp className="h-4 w-4" />}
        />
        <StatCard
          label="P&L Realizado"
          value={formatCurrency(summary?.total_realized_pnl ?? 0)}
          positive={summary ? summary.total_realized_pnl >= 0 : undefined}
          icon={<TrendingDown className="h-4 w-4" />}
        />
        <StatCard
          label="Exposição"
          value={formatPercent(summary?.exposure_pct ?? 0)}
          change={`${summary?.open_positions ?? 0} positions`}
          icon={<PieChart className="h-4 w-4" />}
        />
      </div>

      {/* Risk Status */}
      <Card>
        <CardHeader>
          <CardTitle>Gestão de Risco</CardTitle>
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

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="flex items-center gap-3 rounded-lg bg-[var(--color-background)] p-3">
            <Shield className="h-5 w-5 text-[var(--color-text-muted)]" />
            <div>
              <p className="text-xs text-[var(--color-text-muted)]">Pode Operar</p>
              <p className={cn("text-sm font-medium", riskStatus?.can_trade ? "text-green-400" : "text-red-400")}>
                {riskStatus?.can_trade ? "Sim" : "Não"}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3 rounded-lg bg-[var(--color-background)] p-3">
            <AlertTriangle className="h-5 w-5 text-[var(--color-text-muted)]" />
            <div>
              <p className="text-xs text-[var(--color-text-muted)]">Drawdown Diário</p>
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
              <p className="text-xs text-[var(--color-text-muted)]">Drawdown Semanal</p>
              <p className="text-sm font-mono font-medium">
                {formatPercent(riskStatus?.weekly_drawdown ?? 0)}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3 rounded-lg bg-[var(--color-background)] p-3">
            <TrendingDown className="h-5 w-5 text-[var(--color-text-muted)]" />
            <div>
              <p className="text-xs text-[var(--color-text-muted)]">Mult. Tamanho Posição</p>
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
          <CardTitle>Posições Abertas</CardTitle>
          <Badge variant="primary">{positions.length}</Badge>
        </CardHeader>

        <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Symbol</TableHead>
              <TableHead>Modo</TableHead>
              <TableHead>Side</TableHead>
              <TableHead>Quantidade</TableHead>
              <TableHead>Preço de Entrada</TableHead>
              <TableHead>Preço Atual</TableHead>
              <TableHead>P&L Não Realizado</TableHead>
              <TableHead>Stop Loss</TableHead>
              <TableHead>Take Profit</TableHead>
              <TableHead>Proteção</TableHead>
              <TableHead>Abertura</TableHead>
              <TableHead>Ação</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {positions.length === 0 ? (
              <TableRow>
                <TableCell colSpan={12} className="py-12 text-center text-[var(--color-text-muted)]">
                  Sem posições abertas. A IA abrirá operações quando detectar sinais fortes.
                </TableCell>
              </TableRow>
            ) : (
              positions.map((pos) => (
                <TableRow key={pos.id}>
                  <TableCell className="font-semibold">{pos.symbol}</TableCell>
                  <TableCell>
                    <Badge variant={pos.execution_mode === "LIVE" ? "danger" : "warning"}>
                      {pos.execution_mode}
                    </Badge>
                  </TableCell>
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
                  <TableCell>
                    <Badge
                      variant={
                        pos.protection_status === "ACTIVE"
                          ? "success"
                          : pos.protection_status === "MISSING"
                            ? "danger"
                            : "warning"
                      }
                    >
                      {pos.protection_status === "ACTIVE"
                        ? `OCO #${pos.protective_order_list_id}`
                        : pos.protection_status}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-xs text-[var(--color-text-muted)]">
                    {timeAgo(pos.opened_at)}
                  </TableCell>
                  <TableCell>
                    {pos.execution_mode === "LIVE" && pos.protection_status === "ACTIVE" ? (
                      <Button variant="danger" size="sm" onClick={() => setPositionToClose(pos)}>
                        Encerrar na Binance
                      </Button>
                    ) : (
                      <span className="text-xs text-[var(--color-text-muted)]">
                        {pos.execution_mode === "LIVE" ? "Intervenção manual" : "—"}
                      </span>
                    )}
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
        </div>
      </Card>

      <Dialog
        open={positionToClose !== null}
        onClose={resetCloseDialog}
        title={positionToClose ? `Encerrar ${positionToClose.symbol} na Binance` : undefined}
        description="A OCO ativa será cancelada, conferida por leitura assinada e só então a posição será vendida. Se a OCO já tiver saído, nenhuma segunda venda será enviada."
      >
        {positionToClose && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3 rounded-lg bg-[var(--color-background)] p-3 text-sm">
              <div>
                <p className="text-xs text-[var(--color-text-muted)]">Quantidade</p>
                <p className="mt-1 font-mono">{positionToClose.quantity.toFixed(6)}</p>
              </div>
              <div>
                <p className="text-xs text-[var(--color-text-muted)]">OCO ativa</p>
                <p className="mt-1 font-mono">#{positionToClose.protective_order_list_id}</p>
              </div>
            </div>
            <Input
              label="Código TOTP"
              inputMode="numeric"
              autoComplete="one-time-code"
              value={closeTotpCode}
              onChange={(event) => setCloseTotpCode(event.target.value.replace(/\D/g, "").slice(0, 6))}
              placeholder="000000"
            />
            <Input
              label="Confirmação"
              value={closeConfirmation}
              onChange={(event) => setCloseConfirmation(event.target.value)}
              autoComplete="off"
              placeholder="Digite CLOSE SPOT POSITION"
            />
            {closeError && <p className="text-sm text-red-400">{closeError}</p>}
            <DialogFooter>
              <Button variant="ghost" onClick={resetCloseDialog} disabled={closeLoading}>
                Cancelar
              </Button>
              <Button
                variant="danger"
                onClick={closePositionOnExchange}
                disabled={
                  closeLoading
                  || closeTotpCode.length !== 6
                  || closeConfirmation !== "CLOSE SPOT POSITION"
                }
              >
                {closeLoading ? "Encerrando na Binance..." : "Confirmar saída real"}
              </Button>
            </DialogFooter>
          </div>
        )}
      </Dialog>
    </div>
  );
}
