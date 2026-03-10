"use client";

import { useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { PageHeader } from "@/components/ui/page-header";
import { Spinner } from "@/components/ui/progress";
import { apiFetch } from "@/lib/utils";

interface SentimentData {
  fear_greed_index: number;
  fear_greed_label: string;
  funding_rates: Record<string, number>;
  long_short_ratio: Record<string, number>;
  open_interest: Record<string, number>;
}

export default function SentimentPage() {
  const [data, setData] = useState<SentimentData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchSentiment();
    const interval = setInterval(fetchSentiment, 60000);
    return () => clearInterval(interval);
  }, []);

  async function fetchSentiment() {
    try {
      const result = await apiFetch<SentimentData>("/api/v1/market/sentiment");
      setData(result);
    } catch {} finally { setLoading(false); }
  }

  function getFearGreedColor(value: number) {
    if (value <= 25) return "text-[var(--color-danger)]";
    if (value <= 45) return "text-[var(--color-warning)]";
    if (value <= 55) return "text-[var(--color-text-muted)]";
    if (value <= 75) return "text-[var(--color-success)]";
    return "text-[var(--color-success)]";
  }

  return (
    <div className="space-y-6">
      <PageHeader title="Sentimento de Mercado" description="Indicadores de sentimento de mercado em tempo real" />

      {loading ? (
        <div className="flex justify-center py-12"><Spinner size="lg" /></div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Fear & Greed Index */}
          <Card className="md:col-span-2 lg:col-span-1">
            <CardContent>
              <h2 className="text-xs font-medium uppercase tracking-wider text-[var(--color-text-muted)] mb-4">Índice Medo & Ganância</h2>
              <div className="flex flex-col items-center">
                <div className={`text-6xl font-bold tabular-nums ${getFearGreedColor(data?.fear_greed_index ?? 50)}`}>
                  {data?.fear_greed_index ?? "--"}
                </div>
                <div className="mt-2 text-lg text-[var(--color-text)]">{data?.fear_greed_label ?? "Neutro"}</div>
                <div className="mt-4 w-full">
                  <div className="h-3 w-full overflow-hidden rounded-full bg-[var(--color-background)]">
                    <div
                      className="h-3 rounded-full bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 transition-all duration-500"
                      style={{ width: `${data?.fear_greed_index ?? 50}%` }}
                    />
                  </div>
                  <div className="mt-1 flex justify-between text-xs text-[var(--color-text-muted)]">
                    <span>Medo Extremo</span>
                    <span>Ganância Extrema</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Funding Rates */}
          <Card>
            <CardContent>
              <h2 className="text-xs font-medium uppercase tracking-wider text-[var(--color-text-muted)] mb-4">Taxas de Funding</h2>
              <div className="space-y-4">
                {data?.funding_rates ? Object.entries(data.funding_rates).map(([symbol, rate]) => (
                  <div key={symbol} className="flex justify-between items-center">
                    <span className="font-medium text-[var(--color-text)]">{symbol}</span>
                    <span className={`tabular-nums font-mono text-sm ${rate >= 0 ? "text-[var(--color-success)]" : "text-[var(--color-danger)]"}`}>
                      {(rate * 100).toFixed(4)}%
                    </span>
                  </div>
                )) : (
                  <p className="text-sm text-[var(--color-text-muted)]">Dados indisponíveis</p>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Long/Short Ratio */}
          <Card>
            <CardContent>
              <h2 className="text-xs font-medium uppercase tracking-wider text-[var(--color-text-muted)] mb-4">Proporção Long/Short</h2>
              <div className="space-y-4">
                {data?.long_short_ratio ? Object.entries(data.long_short_ratio).map(([symbol, ratio]) => {
                  const longPct = (ratio / (ratio + 1)) * 100;
                  const shortPct = (1 / (ratio + 1)) * 100;
                  return (
                    <div key={symbol}>
                      <div className="mb-1 flex justify-between text-sm">
                        <span className="text-[var(--color-text)]">{symbol}</span>
                        <span className="tabular-nums text-[var(--color-text-muted)]">{ratio.toFixed(2)}</span>
                      </div>
                      <div className="flex h-2 overflow-hidden rounded-full">
                        <div className="bg-[var(--color-success)]" style={{ width: `${longPct}%` }} />
                        <div className="flex-1 bg-[var(--color-danger)]" />
                      </div>
                      <div className="mt-1 flex justify-between text-xs text-[var(--color-text-muted)]">
                        <span>Long {longPct.toFixed(1)}%</span>
                        <span>Short {shortPct.toFixed(1)}%</span>
                      </div>
                    </div>
                  );
                }) : (
                  <p className="text-sm text-[var(--color-text-muted)]">Dados indisponíveis</p>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Open Interest */}
          <Card>
            <CardContent>
              <h2 className="text-xs font-medium uppercase tracking-wider text-[var(--color-text-muted)] mb-4">Contratos em Aberto</h2>
              <div className="space-y-4">
                {data?.open_interest ? Object.entries(data.open_interest).map(([symbol, oi]) => (
                  <div key={symbol} className="flex justify-between items-center">
                    <span className="font-medium text-[var(--color-text)]">{symbol}</span>
                    <span className="tabular-nums font-mono text-sm text-[var(--color-text-muted)]">${(oi / 1e9).toFixed(2)}B</span>
                  </div>
                )) : (
                  <p className="text-sm text-[var(--color-text-muted)]">Dados indisponíveis</p>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Market Overview */}
          <Card>
            <CardContent>
              <h2 className="text-xs font-medium uppercase tracking-wider text-[var(--color-text-muted)] mb-4">Visão do Mercado</h2>
              <div className="space-y-3">
                {[
                  { label: "Dominância BTC", value: "54.2%" },
                  { label: "Dominância ETH", value: "17.8%" },
                  { label: "Cap. Total de Mercado", value: "$3.2T" },
                  { label: "Volume 24h", value: "$89.4B" },
                ].map(({ label, value }) => (
                  <div key={label} className="flex justify-between">
                    <span className="text-sm text-[var(--color-text-muted)]">{label}</span>
                    <span className="text-sm font-medium tabular-nums text-[var(--color-text)]">{value}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Sentiment Summary */}
          <Card>
            <CardContent>
              <h2 className="text-xs font-medium uppercase tracking-wider text-[var(--color-text-muted)] mb-4">Resumo do Sentimento</h2>
              <div className="space-y-3">
                <div className="rounded-[var(--radius-md)] bg-[var(--color-background)] p-3">
                  <span className="text-xs text-[var(--color-text-muted)]">Viés Geral</span>
                  <p className="text-lg font-semibold text-[var(--color-warning)]">Neutro</p>
                </div>
                <p className="text-sm text-[var(--color-text-muted)]">
                  O sentimento do mercado está misto. As taxas de funding sugerem posicionamento equilibrado.
                  Monitore mudanças direcionais nas proporções long/short.
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
