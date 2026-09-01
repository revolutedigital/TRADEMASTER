"use client";

import { useEffect, useState } from "react";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { StatCard } from "@/components/ui/stat-card";
import { apiFetch, formatNumber, formatPercent } from "@/lib/utils";
import type { BacktestResult, StrategyDeployment } from "@/lib/types";
import { Sliders, Play, Loader2, BarChart3 } from "lucide-react";

interface IndicatorConfig {
  id: string;
  label: string;
  enabled: boolean;
  params: {
    label: string;
    key: string;
    value: number;
    min: number;
    max: number;
    step: number;
  }[];
}

type ExecutionMode = "PAPER" | "TESTNET" | "LIVE";

interface RuntimeTradingStatus {
  execution_mode: ExecutionMode;
}

function isExecutionMode(value: unknown): value is ExecutionMode {
  return value === "PAPER" || value === "TESTNET" || value === "LIVE";
}

const defaultIndicators: IndicatorConfig[] = [
  {
    id: "sma",
    label: "SMA (Simple Moving Average)",
    enabled: false,
    params: [
      {
        label: "Período Curto",
        key: "sma_short",
        value: 10,
        min: 2,
        max: 100,
        step: 1,
      },
      {
        label: "Período Longo",
        key: "sma_long",
        value: 30,
        min: 5,
        max: 200,
        step: 1,
      },
    ],
  },
  {
    id: "ema",
    label: "EMA (Exponential Moving Average)",
    enabled: false,
    params: [
      {
        label: "Período Curto",
        key: "ema_short",
        value: 12,
        min: 2,
        max: 100,
        step: 1,
      },
      {
        label: "Período Longo",
        key: "ema_long",
        value: 26,
        min: 5,
        max: 200,
        step: 1,
      },
    ],
  },
  {
    id: "rsi",
    label: "RSI (Relative Strength Index)",
    enabled: false,
    params: [
      {
        label: "Período",
        key: "rsi_period",
        value: 14,
        min: 2,
        max: 50,
        step: 1,
      },
      {
        label: "Sobrecomprado",
        key: "rsi_overbought",
        value: 70,
        min: 50,
        max: 95,
        step: 1,
      },
      {
        label: "Sobrevendido",
        key: "rsi_oversold",
        value: 30,
        min: 5,
        max: 50,
        step: 1,
      },
    ],
  },
  {
    id: "macd",
    label: "MACD",
    enabled: false,
    params: [
      {
        label: "Período Rápido",
        key: "macd_fast",
        value: 12,
        min: 2,
        max: 50,
        step: 1,
      },
      {
        label: "Período Lento",
        key: "macd_slow",
        value: 26,
        min: 5,
        max: 100,
        step: 1,
      },
      {
        label: "Período do Sinal",
        key: "macd_signal",
        value: 9,
        min: 2,
        max: 50,
        step: 1,
      },
    ],
  },
  {
    id: "bollinger",
    label: "Bollinger Bands",
    enabled: false,
    params: [
      {
        label: "Período",
        key: "bb_period",
        value: 20,
        min: 5,
        max: 100,
        step: 1,
      },
      {
        label: "Desvio Padrão",
        key: "bb_std",
        value: 2,
        min: 0.5,
        max: 4,
        step: 0.1,
      },
    ],
  },
  {
    id: "engulfing",
    label: "Padrão de Engolfo (Reversão)",
    enabled: false,
    params: [],
  },
  {
    id: "breakout",
    label: "Rompimento de Faixa",
    enabled: false,
    params: [
      {
        label: "Janela de Rompimento",
        key: "breakout_lookback",
        value: 20,
        min: 5,
        max: 200,
        step: 1,
      },
    ],
  },
];

export default function StrategyBuilderPage() {
  const [indicators, setIndicators] =
    useState<IndicatorConfig[]>(defaultIndicators);
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [interval, setInterval] = useState("1h");
  const [capital, setCapital] = useState(10000);
  const [minConfirmations, setMinConfirmations] = useState(1);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [deployment, setDeployment] = useState<StrategyDeployment | null>(null);
  const [runtimeMode, setRuntimeMode] = useState<ExecutionMode | null>(null);
  const [runtimeModeError, setRuntimeModeError] = useState<string | null>(null);
  const [deploymentLoading, setDeploymentLoading] = useState(false);
  const [activationPhrase, setActivationPhrase] = useState("");
  const [totpCode, setTotpCode] = useState("");

  const enabledIndicators = indicators.filter((ind) => ind.enabled);

  useEffect(() => {
    const requestedSymbol = new URLSearchParams(window.location.search).get("symbol");
    if (requestedSymbol === "BTCUSDT" || requestedSymbol === "ETHUSDT") {
      setSymbol(requestedSymbol);
    }
  }, []);

  useEffect(() => {
    let active = true;

    const loadRuntimeMode = async () => {
      try {
        const status = await apiFetch<RuntimeTradingStatus>(
          "/api/v1/trading/live/status",
        );
        if (!isExecutionMode(status.execution_mode)) {
          throw new Error(
            "O servidor não informou um modo de execução válido.",
          );
        }
        if (active) {
          setRuntimeMode(status.execution_mode);
          setRuntimeModeError(null);
        }
      } catch (err) {
        if (active) {
          setRuntimeModeError(
            err instanceof Error
              ? err.message
              : "Não foi possível conferir o modo de execução do servidor.",
          );
        }
      }
    };

    void loadRuntimeMode();
    return () => {
      active = false;
    };
  }, []);

  const toggleIndicator = (id: string) => {
    setIndicators((prev) =>
      prev.map((ind) =>
        ind.id === id ? { ...ind, enabled: !ind.enabled } : ind,
      ),
    );
  };

  const updateParam = (
    indicatorId: string,
    paramKey: string,
    value: number,
  ) => {
    setIndicators((prev) =>
      prev.map((ind) =>
        ind.id === indicatorId
          ? {
              ...ind,
              params: ind.params.map((p) =>
                p.key === paramKey ? { ...p, value } : p,
              ),
            }
          : ind,
      ),
    );
  };

  const runBacktest = async () => {
    if (enabledIndicators.length === 0) {
      setError("Selecione pelo menos um indicador.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    setDeployment(null);
    setActivationPhrase("");
    setTotpCode("");

    try {
      // Build indicator params payload
      const indicatorParams: Record<string, unknown> = {};
      for (const ind of enabledIndicators) {
        indicatorParams[ind.id] = {};
        for (const p of ind.params) {
          (indicatorParams[ind.id] as Record<string, number>)[p.key] = p.value;
        }
      }

      const data = await apiFetch<BacktestResult>("/api/v1/backtest/run", {
        method: "POST",
        body: JSON.stringify({
          symbol,
          interval,
          initial_capital: capital,
          strategy: {
            kind: "technical_ensemble",
            indicators: Object.keys(indicatorParams),
            indicator_params: indicatorParams,
            min_confirmations: Math.min(
              minConfirmations,
              enabledIndicators.length,
            ),
            execution_profile: "spot_long_only",
          },
        }),
      });

      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Falha no backtest");
    } finally {
      setLoading(false);
    }
  };

  const validateDeployment = async () => {
    if (!result?.id) {
      setError(
        "Este resultado não tem um identificador persistido para validação.",
      );
      return;
    }
    if (!runtimeMode) {
      setError(
        "Confira o modo de execução do servidor antes de validar para o motor.",
      );
      return;
    }

    setDeploymentLoading(true);
    setError(null);
    try {
      const data = await apiFetch<StrategyDeployment>(
        "/api/v1/strategy-deployments",
        {
          method: "POST",
          body: JSON.stringify({
            source_backtest_id: result.id,
            target_execution_mode: runtimeMode,
          }),
        },
      );
      setDeployment(data);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Falha ao validar a estratégia",
      );
    } finally {
      setDeploymentLoading(false);
    }
  };

  const activateDeployment = async () => {
    if (!deployment) return;
    if (runtimeMode !== deployment.target_execution_mode) {
      setError(
        "O ambiente atual do motor não corresponde à estratégia aprovada.",
      );
      return;
    }

    setDeploymentLoading(true);
    setError(null);
    try {
      const data = await apiFetch<StrategyDeployment>(
        `/api/v1/strategy-deployments/${deployment.id}/activate`,
        {
          method: "POST",
          body: JSON.stringify({
            confirmation_phrase: activationPhrase,
            ...(deployment.target_execution_mode === "LIVE"
              ? { totp_code: totpCode }
              : {}),
          }),
        },
      );
      setDeployment(data);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Falha ao ativar a estratégia",
      );
    } finally {
      setDeploymentLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Sliders className="h-6 w-6 text-[var(--color-primary)]" />
        <h1 className="text-2xl font-bold">Criador de Estratégias</h1>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column - Configuration */}
        <div className="space-y-6">
          {/* General Settings */}
          <Card>
            <CardHeader>
              <CardTitle>Configurações Gerais</CardTitle>
            </CardHeader>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div>
                <label className="mb-1 block text-xs text-[var(--color-text-muted)]">
                  Par
                </label>
                <select
                  value={symbol}
                  onChange={(e) => setSymbol(e.target.value)}
                  className="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm text-[var(--color-text)]"
                >
                  <option value="BTCUSDT">BTC/USDT</option>
                  <option value="ETHUSDT">ETH/USDT</option>
                </select>
              </div>

              <div>
                <label className="mb-1 block text-xs text-[var(--color-text-muted)]">
                  Intervalo
                </label>
                <select
                  value={interval}
                  onChange={(e) => setInterval(e.target.value)}
                  className="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm text-[var(--color-text)]"
                >
                  <option value="15m">15 minutos</option>
                  <option value="1h">1 hora</option>
                  <option value="4h">4 horas</option>
                </select>
                <p className="mt-1 text-xs text-[var(--color-text-muted)]">
                  Intervalos aceitos pelo motor: 15m, 1h e 4h.
                </p>
              </div>

              <div>
                <label className="mb-1 block text-xs text-[var(--color-text-muted)]">
                  Capital Simulado (USDT)
                </label>
                <input
                  type="number"
                  value={capital}
                  onChange={(e) => setCapital(Number(e.target.value))}
                  className="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm text-[var(--color-text)]"
                />
              </div>

              <div>
                <label className="mb-1 block text-xs text-[var(--color-text-muted)]">
                  Confirmações mínimas
                </label>
                <input
                  type="number"
                  value={minConfirmations}
                  min={1}
                  max={Math.max(1, enabledIndicators.length)}
                  onChange={(e) =>
                    setMinConfirmations(Math.max(1, Number(e.target.value)))
                  }
                  className="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm text-[var(--color-text)]"
                />
              </div>
            </div>
          </Card>

          {/* Indicators */}
          <Card>
            <CardHeader>
              <CardTitle>Indicadores</CardTitle>
              <Badge variant="primary">
                {enabledIndicators.length} selecionados
              </Badge>
            </CardHeader>

            <div className="space-y-4">
              {indicators.map((ind) => (
                <div
                  key={ind.id}
                  className="rounded-lg border border-[var(--color-border)] p-3"
                >
                  <label className="flex items-center gap-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={ind.enabled}
                      onChange={() => toggleIndicator(ind.id)}
                      className="h-4 w-4 rounded border-[var(--color-border)] accent-[var(--color-primary)]"
                    />
                    <span className="text-sm font-medium">{ind.label}</span>
                  </label>

                  {ind.enabled && (
                    <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-3 pl-7">
                      {ind.params.map((param) => (
                        <div key={param.key}>
                          <label className="mb-1 block text-xs text-[var(--color-text-muted)]">
                            {param.label}
                          </label>
                          <input
                            type="number"
                            value={param.value}
                            min={param.min}
                            max={param.max}
                            step={param.step}
                            onChange={(e) =>
                              updateParam(
                                ind.id,
                                param.key,
                                Number(e.target.value),
                              )
                            }
                            className="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-1.5 text-sm text-[var(--color-text)]"
                          />
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* Right Column - Preview & Results */}
        <div className="space-y-6">
          {/* Preview */}
          <Card>
            <CardHeader>
              <CardTitle>Prévia da Estratégia</CardTitle>
            </CardHeader>

            <div className="space-y-3">
              <div className="flex items-center justify-between text-sm">
                <span className="text-[var(--color-text-muted)]">Par</span>
                <span className="font-medium">
                  {symbol.replace("USDT", "/USDT")}
                </span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-[var(--color-text-muted)]">
                  Intervalo
                </span>
                <span className="font-medium">{interval}</span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-[var(--color-text-muted)]">Capital</span>
                <span className="font-medium font-mono">
                  {formatNumber(capital)} USDT
                </span>
              </div>

              <div className="border-t border-[var(--color-border)] pt-3">
                <p className="mb-2 text-xs text-[var(--color-text-muted)]">
                  Indicadores Selecionados
                </p>
                {enabledIndicators.length === 0 ? (
                  <p className="text-sm text-[var(--color-text-muted)]">
                    Nenhum indicador selecionado. Escolha pelo menos um para
                    rodar um backtest.
                  </p>
                ) : (
                  <div className="flex flex-wrap gap-2">
                    {enabledIndicators.map((ind) => (
                      <Badge key={ind.id} variant="primary">
                        {ind.id.toUpperCase()}
                      </Badge>
                    ))}
                  </div>
                )}
              </div>

              <p className="border-t border-[var(--color-border)] pt-3 text-xs text-[var(--color-text-muted)]">
                Pesquisa histórica: sinais confirmados no fechamento entram na
                abertura da vela seguinte. No perfil Spot, sinais de venda
                fecham uma posição comprada; nunca abrem short.
              </p>
            </div>

            <div className="mt-4">
              <Button
                variant="primary"
                size="lg"
                className="w-full"
                onClick={runBacktest}
                disabled={loading || enabledIndicators.length === 0}
              >
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Rodando Backtest...
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" />
                    Rodar Backtest
                  </>
                )}
              </Button>
            </div>
          </Card>

          {/* Error */}
          {error && (
            <Card className="border-red-500/50 bg-red-500/10">
              <p className="text-sm text-red-400">{error}</p>
            </Card>
          )}

          {/* Results */}
          {result && (
            <>
              <Card>
                <CardHeader>
                  <CardTitle>
                    <div className="flex items-center gap-2">
                      <BarChart3 className="h-4 w-4" />
                      Resultados do Backtest
                    </div>
                  </CardTitle>
                  <Badge
                    variant={result.total_return >= 0 ? "success" : "danger"}
                  >
                    {result.total_trades} trades
                  </Badge>
                </CardHeader>

                <div className="grid grid-cols-2 gap-3">
                  <StatCard
                    label="Retorno Total"
                    value={`${formatNumber(result.total_return)} USDT`}
                    change={formatPercent(result.total_return_pct)}
                    positive={result.total_return >= 0}
                  />
                  <StatCard
                    label="Taxa de Acerto"
                    value={formatPercent(result.win_rate)}
                    change={`${result.winning_trades}W / ${result.losing_trades}L`}
                    positive={result.win_rate > 0.5}
                  />
                  <StatCard
                    label="Fator de Lucro"
                    value={result.profit_factor.toFixed(2)}
                    positive={result.profit_factor > 1}
                  />
                  <StatCard
                    label="Sharpe Ratio"
                    value={result.sharpe_ratio.toFixed(2)}
                    positive={result.sharpe_ratio > 1}
                  />
                  <StatCard
                    label="Drawdown Máximo"
                    value={formatPercent(result.max_drawdown_pct)}
                    positive={false}
                  />
                  <StatCard
                    label="Total de Trades"
                    value={String(result.total_trades)}
                    positive={result.total_trades > 0}
                  />
                </div>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Validação para o motor</CardTitle>
                  {deployment && (
                    <Badge
                      variant={
                        deployment.status === "REJECTED"
                          ? "danger"
                          : deployment.status === "ACTIVE"
                            ? "success"
                            : "primary"
                      }
                    >
                      {deployment.status}
                    </Badge>
                  )}
                </CardHeader>

                <div className="space-y-3">
                  <p className="text-xs text-[var(--color-text-muted)]">
                    Reexecuta a estratégia com dados recentes e só aprova se o
                    backtest e o walk-forward atenderem os critérios de risco.
                    Isso não envia ordens.
                  </p>

                  <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] p-3">
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <p className="text-xs text-[var(--color-text-muted)]">
                          Ambiente efetivo do motor
                        </p>
                        <p className="mt-1 text-sm">
                          A validação fica vinculada ao ambiente que o servidor
                          está usando agora.
                        </p>
                      </div>
                      <Badge
                        variant={runtimeMode === "LIVE" ? "danger" : "warning"}
                      >
                        {runtimeMode ?? "CONFERINDO"}
                      </Badge>
                    </div>
                    {runtimeModeError && (
                      <p className="mt-2 text-xs text-red-400">
                        {runtimeModeError}
                      </p>
                    )}
                  </div>

                  {!deployment && (
                    <Button
                      variant="primary"
                      className="w-full"
                      onClick={validateDeployment}
                      disabled={deploymentLoading || !result.id || !runtimeMode}
                    >
                      {deploymentLoading
                        ? "Validando dados recentes..."
                        : "Validar para o motor"}
                    </Button>
                  )}

                  {deployment && (
                    <>
                      <div className="grid grid-cols-2 gap-3 text-sm">
                        <div>
                          <span className="block text-xs text-[var(--color-text-muted)]">
                            Janelas fora da amostra
                          </span>
                          <span className="font-medium">
                            {deployment.walk_forward_windows}
                          </span>
                        </div>
                        <div>
                          <span className="block text-xs text-[var(--color-text-muted)]">
                            Trades fora da amostra
                          </span>
                          <span className="font-medium">
                            {deployment.total_test_trades}
                          </span>
                        </div>
                        <div>
                          <span className="block text-xs text-[var(--color-text-muted)]">
                            Retorno médio
                          </span>
                          <span className="font-medium">
                            {formatPercent(deployment.avg_return_pct)}
                          </span>
                        </div>
                        <div>
                          <span className="block text-xs text-[var(--color-text-muted)]">
                            Drawdown médio
                          </span>
                          <span className="font-medium">
                            {formatPercent(deployment.avg_max_drawdown_pct)}
                          </span>
                        </div>
                      </div>

                      {deployment.rejection_reason ? (
                        <p className="rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-xs text-red-300">
                          {deployment.rejection_reason}
                        </p>
                      ) : (
                        <p className="rounded-lg border border-emerald-500/40 bg-emerald-500/10 p-3 text-xs text-emerald-300">
                          Aprovada para {deployment.target_execution_mode}. A
                          ativação exige que o ambiente atual seja o mesmo; em
                          LIVE, o armamento continua separado.
                        </p>
                      )}

                      {deployment.status === "APPROVED" &&
                        runtimeMode !== deployment.target_execution_mode && (
                          <p className="rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-xs text-red-300">
                            O motor está em {runtimeMode ?? "modo desconhecido"}
                            ; esta estratégia foi aprovada para{" "}
                            {deployment.target_execution_mode}. A ativação
                            permanece bloqueada.
                          </p>
                        )}

                      {deployment.status === "APPROVED" && (
                        <div className="space-y-2 border-t border-[var(--color-border)] pt-3">
                          <input
                            value={activationPhrase}
                            onChange={(event) =>
                              setActivationPhrase(event.target.value)
                            }
                            placeholder="Digite ACTIVATE STRATEGY"
                            className="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm text-[var(--color-text)]"
                          />
                          {deployment.target_execution_mode === "LIVE" && (
                            <input
                              value={totpCode}
                              onChange={(event) =>
                                setTotpCode(event.target.value)
                              }
                              inputMode="numeric"
                              maxLength={6}
                              placeholder="Código TOTP de 6 dígitos"
                              className="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm text-[var(--color-text)]"
                            />
                          )}
                          <Button
                            variant="primary"
                            className="w-full"
                            onClick={activateDeployment}
                            disabled={
                              deploymentLoading ||
                              activationPhrase !== "ACTIVATE STRATEGY" ||
                              runtimeMode !== deployment.target_execution_mode
                            }
                          >
                            {deploymentLoading
                              ? "Ativando..."
                              : "Ativar estratégia aprovada"}
                          </Button>
                        </div>
                      )}
                    </>
                  )}
                </div>
              </Card>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
