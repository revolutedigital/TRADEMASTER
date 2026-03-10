"use client";

import { useState, useEffect, useCallback } from "react";
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
import { apiFetch, formatCurrency, timeAgo, cn } from "@/lib/utils";
import type { Order } from "@/lib/types";
import { History, ChevronLeft, ChevronRight, Search } from "lucide-react";

const PAGE_SIZE = 20;

export default function TradeHistoryPage() {
  const [orders, setOrders] = useState<Order[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [symbol, setSymbol] = useState("ALL");
  const [side, setSide] = useState("ALL");
  const [status, setStatus] = useState("ALL");

  // Pagination
  const [page, setPage] = useState(1);

  const fetchOrders = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams();
      if (symbol !== "ALL") params.set("symbol", symbol);
      if (side !== "ALL") params.set("side", side);
      if (status !== "ALL") params.set("status", status);
      if (startDate) params.set("start_date", startDate);
      if (endDate) params.set("end_date", endDate);
      params.set("limit", String(PAGE_SIZE));
      params.set("offset", String((page - 1) * PAGE_SIZE));

      const query = params.toString();
      const data = await apiFetch<Order[]>(
        `/api/v1/trading/orders${query ? `?${query}` : ""}`
      );
      setOrders(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Falha ao buscar ordens");
    } finally {
      setLoading(false);
    }
  }, [symbol, side, status, startDate, endDate, page]);

  useEffect(() => {
    fetchOrders();
  }, [fetchOrders]);

  const handleApplyFilters = () => {
    setPage(1);
    fetchOrders();
  };

  const handleResetFilters = () => {
    setStartDate("");
    setEndDate("");
    setSymbol("ALL");
    setSide("ALL");
    setStatus("ALL");
    setPage(1);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <History className="h-6 w-6 text-[var(--color-primary)]" />
        <h1 className="text-2xl font-bold">Histórico de Operações</h1>
      </div>

      {/* Filters */}
      <Card>
        <CardHeader>
          <CardTitle>Filtros</CardTitle>
        </CardHeader>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          {/* Start Date */}
          <div>
            <label className="mb-1 block text-xs text-[var(--color-text-muted)]">
              Data Inicial
            </label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm text-[var(--color-text)]"
            />
          </div>

          {/* End Date */}
          <div>
            <label className="mb-1 block text-xs text-[var(--color-text-muted)]">
              Data Final
            </label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm text-[var(--color-text)]"
            />
          </div>

          {/* Symbol */}
          <div>
            <label className="mb-1 block text-xs text-[var(--color-text-muted)]">
              Par
            </label>
            <select
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              className="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm text-[var(--color-text)]"
            >
              <option value="ALL">Todos os Pares</option>
              <option value="BTCUSDT">BTC/USDT</option>
              <option value="ETHUSDT">ETH/USDT</option>
            </select>
          </div>

          {/* Side */}
          <div>
            <label className="mb-1 block text-xs text-[var(--color-text-muted)]">
              Lado
            </label>
            <select
              value={side}
              onChange={(e) => setSide(e.target.value)}
              className="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm text-[var(--color-text)]"
            >
              <option value="ALL">Todos</option>
              <option value="BUY">BUY</option>
              <option value="SELL">SELL</option>
            </select>
          </div>

          {/* Status */}
          <div>
            <label className="mb-1 block text-xs text-[var(--color-text-muted)]">
              Status
            </label>
            <select
              value={status}
              onChange={(e) => setStatus(e.target.value)}
              className="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm text-[var(--color-text)]"
            >
              <option value="ALL">Todos</option>
              <option value="FILLED">FILLED</option>
              <option value="CANCELLED">CANCELLED</option>
              <option value="PENDING">PENDING</option>
              <option value="PARTIAL">PARTIAL</option>
            </select>
          </div>
        </div>

        <div className="mt-4 flex items-center gap-2 justify-end">
          <Button variant="ghost" size="sm" onClick={handleResetFilters}>
            Limpar
          </Button>
          <Button variant="primary" size="sm" onClick={handleApplyFilters}>
            <Search className="mr-1.5 h-3.5 w-3.5" />
            Aplicar Filtros
          </Button>
        </div>
      </Card>

      {/* Error */}
      {error && (
        <Card className="border-red-500/50 bg-red-500/10">
          <p className="text-sm text-red-400">{error}</p>
        </Card>
      )}

      {/* Results Table */}
      <Card>
        <CardHeader>
          <CardTitle>Ordens</CardTitle>
          <Badge variant="primary">{orders.length} resultados</Badge>
        </CardHeader>

        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Data</TableHead>
                <TableHead>Par</TableHead>
                <TableHead>Lado</TableHead>
                <TableHead>Qtd</TableHead>
                <TableHead>Preço</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>P&amp;L</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {loading ? (
                <TableRow>
                  <TableCell
                    colSpan={7}
                    className="py-12 text-center text-[var(--color-text-muted)]"
                  >
                    Carregando ordens...
                  </TableCell>
                </TableRow>
              ) : orders.length === 0 ? (
                <TableRow>
                  <TableCell
                    colSpan={7}
                    className="py-12 text-center text-[var(--color-text-muted)]"
                  >
                    Nenhuma ordem encontrada com os filtros atuais.
                  </TableCell>
                </TableRow>
              ) : (
                orders.map((order) => (
                  <TableRow key={order.id}>
                    <TableCell className="text-xs text-[var(--color-text-muted)]">
                      {timeAgo(order.created_at)}
                    </TableCell>
                    <TableCell className="font-medium">
                      {order.symbol}
                    </TableCell>
                    <TableCell>
                      <Badge
                        variant={order.side === "BUY" ? "success" : "danger"}
                      >
                        {order.side}
                      </Badge>
                    </TableCell>
                    <TableCell className="font-mono text-xs">
                      {order.quantity.toFixed(6)}
                    </TableCell>
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
                    <TableCell
                      className={cn(
                        "font-mono text-xs font-medium",
                        order.filled_qty > 0
                          ? "text-[var(--color-text)]"
                          : "text-[var(--color-text-muted)]"
                      )}
                    >
                      {order.filled_qty > 0
                        ? formatCurrency(
                            (order.price - order.price) * order.filled_qty
                          )
                        : "-"}
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </div>

        {/* Pagination */}
        {!loading && orders.length > 0 && (
          <div className="mt-4 flex items-center justify-between border-t border-[var(--color-border)] pt-4">
            <span className="text-xs text-[var(--color-text-muted)]">
              Página {page}
            </span>
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page === 1}
              >
                <ChevronLeft className="h-4 w-4" />
                Anterior
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setPage((p) => p + 1)}
                disabled={orders.length < PAGE_SIZE}
              >
                Próximo
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
