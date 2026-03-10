"use client";

import { useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { PageHeader } from "@/components/ui/page-header";
import { Spinner } from "@/components/ui/progress";
import { apiFetch } from "@/lib/utils";

interface FeeData {
  total_fees: number;
  fees_by_symbol: Record<string, number>;
  fee_impact_pct: number;
  monthly_fees: { month: string; amount: number }[];
}

export default function FeesPage() {
  const [data, setData] = useState<FeeData | null>(null);
  const [period, setPeriod] = useState("30d");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchFees();
  }, [period]);

  async function fetchFees() {
    setLoading(true);
    try {
      const result = await apiFetch<FeeData>(`/api/v1/portfolio/fees?period=${period}`);
      setData(result);
    } catch {} finally { setLoading(false); }
  }

  const periods = ["7d", "30d", "90d", "1y"];

  return (
    <div className="space-y-6">
      <PageHeader
        title="Análise de Taxas"
        description="Acompanhe as taxas de trading e seu impacto nos retornos"
        actions={
          <div className="flex gap-1.5 rounded-[var(--radius-md)] bg-[var(--color-background)] p-1">
            {periods.map((p) => (
              <button
                key={p}
                onClick={() => setPeriod(p)}
                className={`rounded-[var(--radius-sm)] px-3 py-1.5 text-xs font-medium transition-colors ${
                  period === p
                    ? "bg-[var(--color-primary)] text-white shadow-sm"
                    : "text-[var(--color-text-muted)] hover:text-[var(--color-text)]"
                }`}
              >
                {p}
              </button>
            ))}
          </div>
        }
      />

      {loading ? (
        <div className="flex justify-center py-12"><Spinner size="lg" /></div>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card>
              <CardContent>
                <h2 className="text-xs font-medium uppercase tracking-wider text-[var(--color-text-muted)]">Total de Taxas Pagas</h2>
                <p className="mt-2 text-3xl font-bold tabular-nums text-[var(--color-text)]">
                  ${data?.total_fees.toFixed(2) ?? "0.00"}
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardContent>
                <h2 className="text-xs font-medium uppercase tracking-wider text-[var(--color-text-muted)]">Impacto nos Retornos</h2>
                <p className="mt-2 text-3xl font-bold tabular-nums text-[var(--color-danger)]">
                  {data?.fee_impact_pct.toFixed(2) ?? "0.00"}%
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardContent>
                <h2 className="text-xs font-medium uppercase tracking-wider text-[var(--color-text-muted)]">Taxa Média por Trade</h2>
                <p className="mt-2 text-3xl font-bold tabular-nums text-[var(--color-warning)]">
                  ${((data?.total_fees ?? 0) / Math.max(Object.keys(data?.fees_by_symbol ?? {}).length, 1)).toFixed(2)}
                </p>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardContent>
              <h2 className="text-xs font-medium uppercase tracking-wider text-[var(--color-text-muted)] mb-4">Taxas por Par</h2>
              <div className="space-y-3">
                {data?.fees_by_symbol && Object.entries(data.fees_by_symbol).sort(([, a], [, b]) => b - a).map(([symbol, fee]) => (
                  <div key={symbol} className="flex items-center gap-4">
                    <span className="w-24 font-medium text-[var(--color-text)]">{symbol}</span>
                    <div className="flex-1 overflow-hidden rounded-full bg-[var(--color-background)] h-2">
                      <div
                        className="h-2 rounded-full bg-[var(--color-warning)] transition-all duration-500"
                        style={{ width: `${(fee / (data.total_fees || 1)) * 100}%` }}
                      />
                    </div>
                    <span className="w-24 text-right tabular-nums text-sm text-[var(--color-text-muted)]">${fee.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
