"use client";

import { useEffect, useMemo, useState } from "react";
import { Brain, ChevronRight, Loader2, Play, Search, ShieldCheck } from "lucide-react";
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
  pattern_study: {
    regime: "UPTREND" | "DOWNTREND" | "RANGE" | "COMPRESSION" | "STRESS";
    pattern: "TREND_CONTINUATION" | "COMPRESSION_BREAKOUT" | "MEAN_REVERSION" | "OBSERVATION_ONLY";
    confidence: number;
    relative_volume: number;
    taker_buy_imbalance: number | null;
    flow_data_available: boolean;
    explanation: string;
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

type AssetStudyJobStatus = "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED" | "INTERRUPTED";

interface AssetStudyJob {
  id: number;
  symbol: string;
  status: AssetStudyJobStatus;
  message: string | null;
  study: AssetStudy | null;
  error_message: string | null;
}

type OpportunityScanStatus = "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED" | "INTERRUPTED";

interface OpportunityCandidate {
  rank: number;
  symbol: string;
  screening_score: number;
  market_trend: AssetStudy["market_study"]["trend"];
  price_change_pct_24h: number;
  quote_volume_24h: number;
  status: "SHORTLISTED" | "STUDYING" | "APPROVED" | "REJECTED" | "UNAVAILABLE" | "FAILED";
  study: AssetStudy | null;
  error_message: string | null;
}

interface OpportunityScan {
  id: number;
  status: OpportunityScanStatus;
  total_assets: number;
  screened_assets: number;
  shortlisted_assets: number;
  studied_assets: number;
  failed_assets: number;
  message: string | null;
  candidates: OpportunityCandidate[];
}

function assetLabel(symbol: string) {
  return symbol.endsWith("USDT") ? `${symbol.slice(0, -4)}/USDT` : symbol;
}

function trendLabel(trend: AssetStudy["market_study"]["trend"]) {
  return trend === "UPTREND" ? "Tendência de alta" : trend === "DOWNTREND" ? "Tendência de baixa" : "Mercado lateral";
}

function patternLabel(pattern: AssetStudy["pattern_study"]["pattern"]) {
  if (pattern === "TREND_CONTINUATION") return "Continuação de tendência";
  if (pattern === "COMPRESSION_BREAKOUT") return "Rompimento de compressão";
  if (pattern === "MEAN_REVERSION") return "Reversão à média";
  return "Somente observação";
}

function executionModeLabel(mode: ExecutionMode | null) {
  if (mode === "PAPER") return "Simulação protegida";
  if (mode === "TESTNET") return "Testnet protegido";
  if (mode === "LIVE") return "Conta real bloqueada";
  return "Confirmando ambiente";
}

function opportunityScanStatusLabel(status: OpportunityScanStatus) {
  if (status === "QUEUED") return "Preparando busca";
  if (status === "RUNNING") return "Buscando oportunidades";
  if (status === "COMPLETED") return "Busca concluída";
  if (status === "FAILED") return "Busca indisponível";
  return "Busca interrompida";
}

function candidateStatusLabel(status: OpportunityCandidate["status"]) {
  if (status === "SHORTLISTED") return "Finalista";
  if (status === "STUDYING") return "Estudando";
  if (status === "APPROVED") return "Estratégia validada";
  if (status === "REJECTED") return "Estratégia não aprovada";
  if (status === "UNAVAILABLE") return "Sem evidência suficiente";
  return "Estudo indisponível";
}

export default function OperarPage() {
  const [assets, setAssets] = useState<Asset[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState("");
  const [executionMode, setExecutionMode] = useState<ExecutionMode | null>(null);
  const [search, setSearch] = useState("");
  const [study, setStudy] = useState<AssetStudy | null>(null);
  const [studyJob, setStudyJob] = useState<AssetStudyJob | null>(null);
  const [opportunityScan, setOpportunityScan] = useState<OpportunityScan | null>(null);
  const [loadingAssets, setLoadingAssets] = useState(true);
  const [studying, setStudying] = useState(false);
  const [scanningMarket, setScanningMarket] = useState(false);
  const [activating, setActivating] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const opportunityScanId = opportunityScan?.id;
  const opportunityScanStatus = opportunityScan?.status;
  const studyJobId = studyJob?.id;
  const studyJobStatus = studyJob?.status;

  useEffect(() => {
    let active = true;
    const requestedSymbol = new URLSearchParams(window.location.search).get("symbol")?.toUpperCase();

    const loadUniverse = async () => {
      try {
        const universe = await apiFetch<AssetUniverse>("/api/v1/asset-intelligence/universe?limit=500");
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

  useEffect(() => {
    if (!opportunityScanId || !opportunityScanStatus || !["QUEUED", "RUNNING"].includes(opportunityScanStatus)) return;

    let active = true;
    const poll = async () => {
      try {
        const nextScan = await apiFetch<OpportunityScan>(
          `/api/v1/asset-intelligence/opportunity-scans/${opportunityScanId}`,
        );
        if (active) setOpportunityScan(nextScan);
      } catch (err) {
        if (active) {
          setMessage(err instanceof Error ? err.message : "Não foi possível acompanhar a busca de oportunidades.");
        }
      }
    };

    const interval = window.setInterval(() => void poll(), 2_500);
    void poll();
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [opportunityScanId, opportunityScanStatus]);

  useEffect(() => {
    if (!studyJobId || !studyJobStatus || !["QUEUED", "RUNNING"].includes(studyJobStatus)) return;

    let active = true;
    const poll = async () => {
      try {
        const nextJob = await apiFetch<AssetStudyJob>(`/api/v1/asset-intelligence/studies/${studyJobId}`);
        if (!active) return;
        setStudyJob(nextJob);
        if (nextJob.status === "COMPLETED" && nextJob.study) {
          setStudy(nextJob.study);
          setMessage(
            nextJob.study.recommendation.deployment_status === "APPROVED"
              ? "Estratégia validada. Ela ainda não opera até você ativar."
              : "O ativo foi estudado, mas não atingiu os critérios de segurança para operar.",
          );
          setStudying(false);
        } else if (nextJob.status === "FAILED" || nextJob.status === "INTERRUPTED") {
          setMessage(nextJob.error_message ?? nextJob.message ?? "O estudo não pôde ser concluído agora.");
          setStudying(false);
        }
      } catch (err) {
        if (active) {
          setMessage(err instanceof Error ? err.message : "Não foi possível acompanhar o estudo.");
          setStudying(false);
        }
      }
    };

    const interval = window.setInterval(() => void poll(), 2_500);
    void poll();
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [studyJobId, studyJobStatus]);

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
      const job = await apiFetch<AssetStudyJob>("/api/v1/asset-intelligence/studies", {
        method: "POST",
        body: JSON.stringify({ symbol: selectedSymbol }),
      });
      setStudyJob(job);
      if (job.status === "COMPLETED" && job.study) {
        setStudy(job.study);
        setStudying(false);
        setMessage("O estudo já estava concluído. Nenhuma estratégia foi ativada.");
      } else if (job.status === "FAILED" || job.status === "INTERRUPTED") {
        setStudying(false);
        setMessage(job.error_message ?? job.message ?? "O estudo não pôde ser concluído agora.");
      } else if (job.symbol !== selectedSymbol) {
        setMessage(`O estudo de ${assetLabel(job.symbol)} já está em andamento. Assim que terminar, você poderá iniciar outro.`);
      } else {
        setMessage("Estudo iniciado em segundo plano. Você pode continuar usando a plataforma enquanto a IA analisa o ativo.");
      }
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Não foi possível concluir o estudo deste ativo.");
      setStudying(false);
    }
  };

  const searchMarketOpportunities = async () => {
    setScanningMarket(true);
    setMessage(null);
    try {
      const scan = await apiFetch<OpportunityScan>("/api/v1/asset-intelligence/opportunity-scans", {
        method: "POST",
      });
      setOpportunityScan(scan);
      setMessage(
        scan.status === "COMPLETED"
          ? "A última busca já está concluída. Escolha um dos ativos estudados abaixo."
          : "A busca está varrendo o mercado. Você pode continuar navegando enquanto ela trabalha.",
      );
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Não foi possível iniciar a busca de oportunidades.");
    } finally {
      setScanningMarket(false);
    }
  };

  const selectOpportunityCandidate = (candidate: OpportunityCandidate) => {
    setSelectedSymbol(candidate.symbol);
    setSearch("");
    setStudy(candidate.study);
    setMessage(
      candidate.study
        ? `${assetLabel(candidate.symbol)} já foi estudado na busca de mercado.`
        : `${assetLabel(candidate.symbol)} selecionado para estudo detalhado.`,
    );
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

  const marketScanIsStudying = opportunityScan?.status === "QUEUED" || opportunityScan?.status === "RUNNING";

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
          <CardTitle>Buscar oportunidades no mercado</CardTitle>
          <Badge variant="default">Todos os ativos líquidos</Badge>
        </CardHeader>
        <div className="flex flex-col gap-3 px-0 pt-0 sm:flex-row sm:items-center sm:justify-between">
          <p className="max-w-2xl text-sm text-[var(--color-text-muted)]">
            Varremos todo o catálogo Spot / USDT e estudamos a fundo apenas os seis finalistas. A busca não ativa estratégia nem envia ordem.
          </p>
          <Button onClick={searchMarketOpportunities} disabled={loadingAssets || scanningMarket || assets.length === 0} className="shrink-0">
            {scanningMarket ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
            {scanningMarket ? "Iniciando busca..." : "Buscar oportunidades no mercado"}
          </Button>
        </div>
      </Card>

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
          <Button onClick={studySelectedAsset} disabled={!selectedSymbol || studying || marketScanIsStudying} className="shrink-0">
            {studying || marketScanIsStudying ? <Loader2 className="h-4 w-4 animate-spin" /> : <Brain className="h-4 w-4" />}
            {marketScanIsStudying ? "Busca ampla estudando..." : studying ? "Estudando em segundo plano..." : `Estudar ${assetLabel(selectedSymbol)}`}
          </Button>
        </div>
      </Card>

      {message && (
        <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-3 text-sm text-[var(--color-text-muted)]">
          {message}
        </div>
      )}

      {opportunityScan && (
        <Card>
          <CardHeader>
            <CardTitle>Oportunidades encontradas</CardTitle>
            <Badge variant={opportunityScan.status === "COMPLETED" ? "success" : opportunityScan.status === "FAILED" || opportunityScan.status === "INTERRUPTED" ? "danger" : "warning"}>
              {opportunityScanStatusLabel(opportunityScan.status)}
            </Badge>
          </CardHeader>
          <div className="space-y-4 px-0 pt-0">
            <div>
              <div className="mb-2 flex items-center justify-between text-xs text-[var(--color-text-muted)]">
                <span>{opportunityScan.message ?? "Preparando a busca."}</span>
                <span>{opportunityScan.screened_assets}/{opportunityScan.total_assets || "—"}</span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-[var(--color-background)]">
                <div
                  className="h-full rounded-full bg-[var(--color-primary)] transition-all"
                  style={{ width: `${opportunityScan.total_assets > 0 ? Math.min(100, (opportunityScan.screened_assets / opportunityScan.total_assets) * 100) : 0}%` }}
                />
              </div>
            </div>

            {opportunityScan.candidates.length > 0 && (
              <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
                {opportunityScan.candidates.map((candidate) => (
                  <div key={candidate.symbol} className="rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] p-3">
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <p className="font-medium">#{candidate.rank} {assetLabel(candidate.symbol)}</p>
                        <p className="mt-1 text-xs text-[var(--color-text-muted)]">
                          {trendLabel(candidate.market_trend)} · pontuação {candidate.screening_score.toFixed(1)}
                        </p>
                      </div>
                      <Badge variant={candidate.status === "APPROVED" ? "success" : candidate.status === "FAILED" ? "danger" : "default"}>
                        {candidateStatusLabel(candidate.status)}
                      </Badge>
                    </div>
                    <p className="mt-3 text-xs text-[var(--color-text-muted)]">
                      Vol. 24h ${formatNumber(candidate.quote_volume_24h, 0)} · {candidate.price_change_pct_24h >= 0 ? "+" : ""}{candidate.price_change_pct_24h.toFixed(2)}%
                    </p>
                    {candidate.error_message && <p className="mt-2 text-xs text-red-400">{candidate.error_message}</p>}
                    <Button className="mt-3 w-full" onClick={() => selectOpportunityCandidate(candidate)}>
                      {candidate.study ? "Abrir estudo" : "Selecionar ativo"}
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </Card>
      )}

      {study && (
        <div className="grid gap-4 lg:grid-cols-4">
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
              <CardTitle>Padrão e fluxo</CardTitle>
              <Badge variant={study.pattern_study.pattern === "OBSERVATION_ONLY" ? "warning" : "success"}>
                {patternLabel(study.pattern_study.pattern)}
              </Badge>
            </CardHeader>
            <div className="space-y-3 px-0 pt-0 text-sm">
              <Metric label="Confiança" value={`${(study.pattern_study.confidence * 100).toFixed(0)}%`} />
              <Metric label="Volume relativo" value={`${study.pattern_study.relative_volume.toFixed(2)}×`} />
              <Metric
                label="Fluxo agressor"
                value={study.pattern_study.taker_buy_imbalance === null ? "Histórico em atualização" : `${(study.pattern_study.taker_buy_imbalance * 100).toFixed(0)}% comprador`}
              />
              <p className="text-xs text-[var(--color-text-muted)]">{study.pattern_study.explanation}</p>
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
        A aprovação não envia ordem. No Testnet, o canário limita a 3 estratégias, 0,25% de risco por entrada, 5% por ativo e 20% de carteira, sempre com stop e alvo nativos. Em conta real, continua bloqueado.
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
