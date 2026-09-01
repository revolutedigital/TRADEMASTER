"use client";

import { useEffect, useMemo, useState } from "react";
import { Brain, CheckCircle2, ChevronRight, Loader2, Play, Search, ShieldCheck } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { apiFetch, formatNumber } from "@/lib/utils";

type ExecutionMode = "PAPER" | "TESTNET" | "LIVE";

interface Asset {
  symbol: string;
  base_asset: string;
  quote_asset: "USDT";
  quote_volume_24h: number;
  price_change_pct_24h: number;
}

interface AssetUniverse {
  assets: Asset[];
}

interface AssetStudy {
  symbol: string;
  execution_mode: ExecutionMode;
  market_study: {
    trend: "UPTREND" | "DOWNTREND" | "RANGE";
    volatility_pct: number;
    liquidity_quote_volume_24h: number;
    candles: number;
  };
  predictive_model: {
    trained: boolean;
    validation_accuracy: number | null;
    samples: number;
    latest_signal: "BUY" | "HOLD" | "SELL" | "UNAVAILABLE";
  };
  recommendation: {
    strategy_name: string;
    backtest_id: number | null;
    deployment_id: number | null;
    deployment_status: "APPROVED" | "REJECTED" | "UNAVAILABLE";
    reasons: string[];
  };
}

function assetLabel(symbol: string) {
  return symbol.endsWith("USDT") ? `${symbol.slice(0, -4)}/USDT` : symbol;
}

function trendLabel(trend: AssetStudy["market_study"]["trend"]) {
  return trend === "UPTREND" ? "Tendência de alta" : trend === "DOWNTREND" ? "Tendência de baixa" : "Mercado lateral";
}

function executionModeLabel(mode: ExecutionMode | null) {
  if (mode === "PAPER") return "Simulação protegida";
  if (mode === "TESTNET") return "Testnet protegido";
  if (mode === "LIVE") return "Conta real bloqueada";
  return "Confirmando ambiente";
}

export default function OperarPage() {
  const [assets, setAssets] = useState<Asset[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState("");
  const [executionMode, setExecutionMode] = useState<ExecutionMode | null>(null);
  const [search, setSearch] = useState("");
  const [study, setStudy] = useState<AssetStudy | null>(null);
  const [loadingAssets, setLoadingAssets] = useState(true);
  const [studying, setStudying] = useState(false);
  const [activating, setActivating] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    const requestedSymbol = new URLSearchParams(window.location.search).get("symbol")?.toUpperCase();

    const loadUniverse = async () => {
      try {
        const universe = await apiFetch<AssetUniverse>("/api/v1/asset-intelligence/universe?limit=250");
        if (!active) return;
        setAssets(universe.assets);
        const preferred = universe.assets.find((asset) => asset.symbol === requestedSymbol)?.symbol;
        setSelectedSymbol(preferred ?? universe.assets[0]?.symbol ?? "");
      } catch (err) {
        if (active) {
          setMessage(err instanceof Error ? err.message : "Não foi possível carregar os ativos disponíveis.");
        }
      } finally {
        if (active) setLoadingAssets(false);
      }

      try {
        const runtime = await apiFetch<{ execution_mode: ExecutionMode }>("/api/v1/trading/live/status");
        if (active) setExecutionMode(runtime.execution_mode);
      } catch {
        // The study response is authoritative for activation; a transient status failure must not hide the asset catalog.
      }
    };

    void loadUniverse();
    return () => {
      active = false;
    };
  }, []);

  const filteredAssets = useMemo(() => {
    const normalized = search.trim().toUpperCase();
    if (!normalized) return assets;
    return assets.filter((asset) => asset.symbol.includes(normalized) || asset.base_asset.includes(normalized));
  }, [assets, search]);

  const selectableAssets = useMemo(() => {
    if (!selectedSymbol || filteredAssets.some((asset) => asset.symbol === selectedSymbol)) {
      return filteredAssets;
    }

    const selectedAsset = assets.find((asset) => asset.symbol === selectedSymbol);
    return selectedAsset ? [selectedAsset, ...filteredAssets] : filteredAssets;
  }, [assets, filteredAssets, selectedSymbol]);

  const studySelectedAsset = async () => {
    if (!selectedSymbol) return;
    setStudying(true);
    setMessage(null);
    setStudy(null);
    try {
      const result = await apiFetch<AssetStudy>("/api/v1/asset-intelligence/studies", {
        method: "POST",
        body: JSON.stringify({ symbol: selectedSymbol }),
      });
      setStudy(result);
      setMessage(
        result.recommendation.deployment_status === "APPROVED"
          ? "Estratégia validada. Ela ainda não opera até você ativar."
          : "O ativo foi estudado, mas não atingiu os critérios de segurança para operar.",
      );
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Não foi possível concluir o estudo deste ativo.");
    } finally {
      setStudying(false);
    }
  };

  const activateRecommendation = async () => {
    if (!study?.recommendation.deployment_id) return;
    setActivating(true);
    setMessage(null);
    try {
      await apiFetch(`/api/v1/strategy-deployments/${study.recommendation.deployment_id}/activate`, {
        method: "POST",
        body: JSON.stringify({ confirmation_phrase: "ACTIVATE STRATEGY" }),
      });
      await apiFetch("/api/v1/trading/engine/start", { method: "POST" });
      setMessage(`Estratégia ativa e acompanhamento iniciado para ${assetLabel(study.symbol)} em ${study.execution_mode}.`);
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Não foi possível ativar o acompanhamento.");
    } finally {
      setActivating(false);
    }
  };

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-sm font-medium text-[var(--color-primary)]">Operação guiada</p>
          <h1 className="text-2xl font-semibold tracking-tight">Escolha um ativo. A IA faz o resto.</h1>
          <p className="mt-1 max-w-2xl text-sm text-[var(--color-text-muted)]">
            O estudo usa dados públicos do mercado, treina um modelo preditivo e só recomenda uma estratégia se o backtest e o walk-forward passarem.
          </p>
        </div>
        <Badge variant={executionMode === "LIVE" ? "danger" : executionMode === "TESTNET" ? "warning" : "default"}>
          {executionModeLabel(executionMode)}
        </Badge>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>1. Escolha o ativo</CardTitle>
          <Badge variant="default">Spot / USDT</Badge>
        </CardHeader>
        <div className="grid gap-3 px-0 pt-0 sm:grid-cols-[minmax(0,1fr)_minmax(0,2fr)]">
          <label className="relative block">
            <Search className="pointer-events-none absolute left-3 top-3.5 h-4 w-4 text-[var(--color-text-faint)]" />
            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Buscar BTC, SOL, DOGE..."
              aria-label="Buscar ativo"
              className="h-10 w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] pl-9 pr-3 text-sm outline-none transition-colors focus:border-[var(--color-primary)]"
            />
          </label>
          <select
            value={selectedSymbol}
            aria-label="Ativo para estudo"
            onChange={(event) => {
              setSelectedSymbol(event.target.value);
              setStudy(null);
              setMessage(null);
            }}
            disabled={loadingAssets || selectableAssets.length === 0}
            className="h-10 rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] px-3 text-sm outline-none transition-colors focus:border-[var(--color-primary)]"
          >
            {selectableAssets.map((asset) => (
              <option key={asset.symbol} value={asset.symbol}>
                {assetLabel(asset.symbol)} · Vol. 24h ${formatNumber(asset.quote_volume_24h, 0)}
              </option>
            ))}
          </select>
        </div>
        <p className="mt-3 text-xs text-[var(--color-text-faint)]">
          Mostramos pares Spot negociáveis, com liquidez mínima. O catálogo não usa sua chave de trade.
        </p>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>2. Estudar e adaptar a estratégia</CardTitle>
          <Brain className="h-5 w-5 text-[var(--color-primary)]" />
        </CardHeader>
        <div className="flex flex-col gap-3 px-0 pt-0 sm:flex-row sm:items-center sm:justify-between">
          <p className="max-w-2xl text-sm text-[var(--color-text-muted)]">
            A IA avalia tendência, volatilidade e liquidez; treina um modelo temporal para este ativo; compara estratégias adequadas ao regime atual e valida a vencedora fora da amostra.
          </p>
          <Button onClick={studySelectedAsset} disabled={!selectedSymbol || studying} className="shrink-0">
            {studying ? <Loader2 className="h-4 w-4 animate-spin" /> : <Brain className="h-4 w-4" />}
            {studying ? "Estudando mercado..." : `Estudar ${assetLabel(selectedSymbol)}`}
          </Button>
        </div>
      </Card>

      {message && (
        <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-3 text-sm text-[var(--color-text-muted)]">
          {message}
        </div>
      )}

      {study && (
        <div className="grid gap-4 lg:grid-cols-3">
          <Card>
            <CardHeader>
              <CardTitle>Leitura do mercado</CardTitle>
              <Badge variant="default">{assetLabel(study.symbol)}</Badge>
            </CardHeader>
            <div className="space-y-3 px-0 pt-0 text-sm">
              <Metric label="Regime" value={trendLabel(study.market_study.trend)} />
              <Metric label="Volatilidade" value={`${study.market_study.volatility_pct.toFixed(2)}%`} />
              <Metric label="Liquidez 24h" value={`$${formatNumber(study.market_study.liquidity_quote_volume_24h, 0)}`} />
            </div>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Modelo preditivo</CardTitle>
              <Brain className="h-5 w-5 text-[var(--color-primary)]" />
            </CardHeader>
            <div className="space-y-3 px-0 pt-0 text-sm">
              <Metric label="Estado" value={study.predictive_model.trained ? "Treinado" : "Indisponível"} />
              <Metric label="Validação temporal" value={study.predictive_model.validation_accuracy === null ? "Sem evidência" : `${(study.predictive_model.validation_accuracy * 100).toFixed(1)}%`} />
              <Metric label="Sinal atual" value={study.predictive_model.latest_signal} />
            </div>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Estratégia recomendada</CardTitle>
              <Badge variant={study.recommendation.deployment_status === "APPROVED" ? "success" : "warning"}>
                {study.recommendation.deployment_status === "APPROVED" ? "Validada" : "Não aprovada"}
              </Badge>
            </CardHeader>
            <div className="space-y-3 px-0 pt-0 text-sm">
              <p className="font-medium">{study.recommendation.strategy_name}</p>
              {study.recommendation.reasons.length > 0 && (
                <ul className="space-y-1 text-xs text-[var(--color-text-muted)]">
                  {study.recommendation.reasons.slice(0, 3).map((reason) => <li key={reason}>• {reason}</li>)}
                </ul>
              )}
              {study.recommendation.deployment_status === "APPROVED" && study.recommendation.deployment_id && study.execution_mode !== "LIVE" && (
                <Button onClick={activateRecommendation} disabled={activating} className="w-full">
                  {activating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                  {activating ? "Ativando..." : "Ativar e acompanhar"}
                </Button>
              )}
              {study.recommendation.deployment_status === "APPROVED" && study.execution_mode === "LIVE" && (
                <p className="rounded-lg border border-yellow-500/30 bg-yellow-500/5 p-3 text-xs text-[var(--color-text-muted)]">
                  A estratégia foi validada, mas a conta real continua bloqueada. A liberação exige a etapa independente de proteção e autorização de operação real.
                </p>
              )}
            </div>
          </Card>
        </div>
      )}

      <div className="flex items-center gap-2 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-3 text-xs text-[var(--color-text-muted)]">
        <ShieldCheck className="h-4 w-4 shrink-0 text-green-400" />
        A aprovação não envia ordem. No Testnet, a estratégia entra no motor com stop e alvo nativos; em conta real, continua bloqueada pelos controles extras de proteção.
        <ChevronRight className="ml-auto h-4 w-4 shrink-0" />
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-[var(--color-border)] pb-2 last:border-0 last:pb-0">
      <span className="text-[var(--color-text-muted)]">{label}</span>
      <span className="font-medium text-right">{value}</span>
    </div>
  );
}
