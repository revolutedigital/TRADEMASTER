"use client";

import { useState, useEffect } from "react";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { apiFetch } from "@/lib/utils";
import type { SystemHealth } from "@/lib/types";
import { Settings, Server, Database, Wifi, ExternalLink } from "lucide-react";

interface RiskConfig {
  max_risk_per_trade: number;
  max_total_exposure: number;
  max_single_asset: number;
  max_daily_drawdown: number;
  max_weekly_drawdown: number;
  max_monthly_drawdown: number;
  max_total_drawdown: number;
  kelly_fraction: number;
}

interface FullSettings {
  trading: {
    trading_mode: string;
    symbols: string[];
    max_risk_per_trade: number;
    max_total_exposure: number;
  };
  risk: RiskConfig;
  api_docs_url: string;
}

interface LiveTradingStatus {
  execution_mode: "PAPER" | "TESTNET" | "LIVE";
  live_enabled: boolean;
  armed: boolean;
  armed_until: string | null;
  armable: boolean;
  blockers: string[];
  max_notional_per_order: number;
  max_daily_notional: number;
  reconciliation: LiveProtectionReadiness;
  testnet_verification: LiveProtectionReadiness;
}

interface LiveProtectionReadiness {
  ready: boolean;
  state: "UNVERIFIED" | "READY" | "UNRESOLVED" | "ERROR" | "STALE";
  checked_at: string | null;
  max_age_seconds: number;
  issues: string[];
}

function getTradingModeCopy(mode: string | undefined) {
  switch (mode) {
    case "live":
      return { label: "Conta Real", description: "Execução real bloqueada até o armamento operacional", variant: "danger" as const };
    case "testnet":
      return { label: "Testnet", description: "Ordens na Binance Testnet, sem capital real", variant: "warning" as const };
    default:
      return { label: "Paper", description: "Ordens simuladas, sem envio à corretora", variant: "warning" as const };
  }
}

export default function SettingsPage() {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [settings, setSettingsData] = useState<FullSettings | null>(null);
  const defaultRisk: RiskConfig = {
    max_risk_per_trade: 0.02,
    max_total_exposure: 0.60,
    max_single_asset: 0.30,
    max_daily_drawdown: 0.03,
    max_weekly_drawdown: 0.07,
    max_monthly_drawdown: 0.10,
    max_total_drawdown: 0.15,
    kelly_fraction: 0.15,
  };
  const [liveTrading, setLiveTrading] = useState<LiveTradingStatus | null>(null);
  const [liveArmCode, setLiveArmCode] = useState("");
  const [liveTotpCode, setLiveTotpCode] = useState("");
  const [liveConfirmation, setLiveConfirmation] = useState("");
  const [testnetConfirmation, setTestnetConfirmation] = useState("");
  const [liveActionLoading, setLiveActionLoading] = useState(false);
  const [liveActionError, setLiveActionError] = useState<string | null>(null);
  const tradingMode = getTradingModeCopy(settings?.trading?.trading_mode);

  useEffect(() => {
    apiFetch<SystemHealth>("/api/v1/system/health")
      .then(setHealth)
      .catch(() => {});
    apiFetch<FullSettings>("/api/v1/settings/")
      .then((s) => {
        setSettingsData(s);
      })
      .catch(() => {});
    let statusRequestActive = true;

    const refreshLiveTradingStatus = () => {
      apiFetch<LiveTradingStatus>("/api/v1/trading/live/status")
        .then((status) => {
          if (statusRequestActive) setLiveTrading(status);
        })
        .catch(() => {
          if (statusRequestActive) setLiveTrading(null);
        });
    };

    refreshLiveTradingStatus();
    const statusRefreshTimer = window.setInterval(refreshLiveTradingStatus, 30_000);

    return () => {
      statusRequestActive = false;
      window.clearInterval(statusRefreshTimer);
    };
  }, []);

  const armLiveTrading = async () => {
    setLiveActionLoading(true);
    setLiveActionError(null);
    try {
      const status = await apiFetch<LiveTradingStatus>("/api/v1/trading/live/arm", {
        method: "POST",
        body: JSON.stringify({
          confirmation_phrase: liveConfirmation,
          arm_code: liveArmCode,
          totp_code: liveTotpCode,
        }),
      });
      setLiveTrading(status);
      setLiveArmCode("");
      setLiveTotpCode("");
      setLiveConfirmation("");
    } catch (err) {
      setLiveActionError(err instanceof Error ? err.message : "Não foi possível armar a conta real");
    } finally {
      setLiveActionLoading(false);
    }
  };

  const disarmLiveTrading = async () => {
    setLiveActionLoading(true);
    setLiveActionError(null);
    try {
      const status = await apiFetch<LiveTradingStatus>("/api/v1/trading/live/disarm", {
        method: "POST",
        body: JSON.stringify({ reason: "operator disarm from settings" }),
      });
      setLiveTrading(status);
    } catch (err) {
      setLiveActionError(err instanceof Error ? err.message : "Não foi possível bloquear a conta real");
    } finally {
      setLiveActionLoading(false);
    }
  };

  const reconcileLiveProtection = async () => {
    setLiveActionLoading(true);
    setLiveActionError(null);
    try {
      await apiFetch<LiveProtectionReadiness>("/api/v1/trading/live/reconcile", {
        method: "POST",
      });
      const status = await apiFetch<LiveTradingStatus>("/api/v1/trading/live/status");
      setLiveTrading(status);
    } catch (err) {
      setLiveActionError(err instanceof Error ? err.message : "Não foi possível reconciliar a proteção OCO");
    } finally {
      setLiveActionLoading(false);
    }
  };

  const verifyTestnetNativeOco = async () => {
    if (!settings?.trading?.symbols[0]) return;
    setLiveActionLoading(true);
    setLiveActionError(null);
    try {
      await apiFetch("/api/v1/trading/live/testnet-protection-verification", {
        method: "POST",
        body: JSON.stringify({
          confirmation_phrase: testnetConfirmation,
          symbol: settings.trading.symbols[0],
        }),
      });
      const status = await apiFetch<LiveTradingStatus>("/api/v1/trading/live/status");
      setLiveTrading(status);
      setTestnetConfirmation("");
    } catch (err) {
      setLiveActionError(err instanceof Error ? err.message : "A verificação Testnet não foi concluída");
    } finally {
      setLiveActionLoading(false);
    }
  };

  const risk = settings?.risk ?? defaultRisk;
  const riskFields: { key: keyof RiskConfig; label: string; min: number; max: number; format: "pct" | "num" }[] = [
    { key: "max_risk_per_trade", label: "Risco Máximo por Trade", min: 0.001, max: 0.10, format: "pct" },
    { key: "max_total_exposure", label: "Exposição Máxima da Carteira", min: 0.01, max: 1.0, format: "pct" },
    { key: "max_single_asset", label: "Máximo por Ativo", min: 0.01, max: 1.0, format: "pct" },
    { key: "max_daily_drawdown", label: "Drawdown Máximo Diário", min: 0.01, max: 0.20, format: "pct" },
    { key: "max_weekly_drawdown", label: "Drawdown Máximo Semanal", min: 0.02, max: 0.30, format: "pct" },
    { key: "max_monthly_drawdown", label: "Drawdown Máximo Mensal", min: 0.03, max: 0.50, format: "pct" },
    { key: "max_total_drawdown", label: "Drawdown Máximo Total", min: 0.05, max: 0.50, format: "pct" },
    { key: "kelly_fraction", label: "Fração de Kelly", min: 0.05, max: 0.50, format: "pct" },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Settings className="h-6 w-6 text-[var(--color-primary)]" />
        <h1 className="text-2xl font-bold">Configurações</h1>
      </div>

      {/* System Health */}
      <Card>
        <CardHeader>
          <CardTitle>Status do Sistema</CardTitle>
          <Badge variant={health?.status === "healthy" ? "success" : "warning"}>
            {health?.status ?? "Loading..."}
          </Badge>
        </CardHeader>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="flex items-center gap-3 rounded-lg bg-[var(--color-background)] p-4">
            <Server className="h-5 w-5 text-[var(--color-text-muted)]" />
            <div>
              <p className="text-xs text-[var(--color-text-muted)]">Servidor API</p>
              <p className="text-sm font-medium text-green-400">
                {health?.services?.api ?? "Desconhecido"}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3 rounded-lg bg-[var(--color-background)] p-4">
            <Database className="h-5 w-5 text-[var(--color-text-muted)]" />
            <div>
              <p className="text-xs text-[var(--color-text-muted)]">Banco de Dados</p>
              <p className="text-sm font-medium">
                {health?.services?.database ?? "Desconhecido"}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3 rounded-lg bg-[var(--color-background)] p-4">
            <Wifi className="h-5 w-5 text-[var(--color-text-muted)]" />
            <div>
              <p className="text-xs text-[var(--color-text-muted)]">Binance WS</p>
              <p className="text-sm font-medium">
                {health?.services?.binance ?? "Desconhecido"}
              </p>
            </div>
          </div>
        </div>

        {health && (
          <p className="mt-3 text-xs text-[var(--color-text-muted)]">
            Versão: {health.version} | Tempo ativo: {Math.floor(health.uptime / 60)}m
          </p>
        )}
      </Card>

      {/* Real Account Safety Control */}
      <Card className={liveTrading?.armed ? "border-red-500/60" : undefined}>
        <CardHeader>
          <CardTitle>Conta Real — Controle de Segurança</CardTitle>
          <Badge variant={liveTrading?.armed ? "danger" : "warning"}>
            {liveTrading?.armed ? "ARMADA" : "BLOQUEADA"}
          </Badge>
        </CardHeader>

        {!liveTrading ? (
          <p className="text-sm text-[var(--color-text-muted)]">Carregando status operacional...</p>
        ) : (
          <div className="space-y-4">
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-4">
              <div className="rounded-lg bg-[var(--color-background)] p-3">
                <p className="text-xs text-[var(--color-text-muted)]">Modo efetivo</p>
                <p className="mt-1 font-medium">{liveTrading.execution_mode}</p>
              </div>
              <div className="rounded-lg bg-[var(--color-background)] p-3">
                <p className="text-xs text-[var(--color-text-muted)]">Teto por ordem</p>
                <p className="mt-1 font-mono text-sm">{liveTrading.max_notional_per_order.toFixed(2)} USDT</p>
              </div>
              <div className="rounded-lg bg-[var(--color-background)] p-3">
                <p className="text-xs text-[var(--color-text-muted)]">Teto diário</p>
                <p className="mt-1 font-mono text-sm">{liveTrading.max_daily_notional.toFixed(2)} USDT</p>
              </div>
              <div className="rounded-lg bg-[var(--color-background)] p-3">
                <p className="text-xs text-[var(--color-text-muted)]">Proteção e inventário</p>
                <Badge variant={liveTrading.reconciliation.ready ? "success" : "warning"}>
                  {liveTrading.reconciliation.state}
                </Badge>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-3 rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] p-3">
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium">Reconciliação nativa da Binance Spot</p>
                <p className="mt-1 text-xs text-[var(--color-text-muted)]">
                  {liveTrading.reconciliation.checked_at
                    ? `Última confirmação: ${new Date(liveTrading.reconciliation.checked_at).toLocaleString("pt-BR")}. Válida por ${liveTrading.reconciliation.max_age_seconds}s.`
                    : "Nenhuma confirmação válida nesta sessão. Conta real permanece bloqueada."}
                </p>
              </div>
              {liveTrading.execution_mode === "LIVE" && !liveTrading.armed && (
                <Button variant="default" size="sm" onClick={reconcileLiveProtection} disabled={liveActionLoading}>
                  {liveActionLoading ? "Conferindo..." : "Conferir OCOs na Binance"}
                </Button>
              )}
            </div>

            <div className="flex flex-col gap-3 rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] p-3 sm:flex-row sm:items-center">
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium">Prova de proteção na Binance Testnet</p>
                <p className="mt-1 text-xs text-[var(--color-text-muted)]">
                  {liveTrading.testnet_verification.checked_at
                    ? `Última prova válida: ${new Date(liveTrading.testnet_verification.checked_at).toLocaleString("pt-BR")}. Válida por ${Math.floor(liveTrading.testnet_verification.max_age_seconds / 86_400)} dias.`
                    : "Obrigatória antes de liberar conta real. A rotina faz uma compra mínima Testnet, cria OCO, confirma, cancela e zera a posição."}
                </p>
              </div>
              <Badge variant={liveTrading.testnet_verification.ready ? "success" : "warning"}>
                {liveTrading.testnet_verification.state}
              </Badge>
            </div>

            {liveTrading.execution_mode === "TESTNET" && !liveTrading.armed && (
              <div className="space-y-3 rounded-lg border border-yellow-500/30 bg-yellow-500/5 p-4">
                <p className="text-sm text-yellow-200">
                  Esta ação opera somente na Testnet: compra o mínimo permitido de {settings?.trading?.symbols[0] ?? "um par configurado"}, valida a OCO nativa e encerra a posição de teste. Ela não habilita conta real.
                </p>
                <input
                  value={testnetConfirmation}
                  onChange={(event) => setTestnetConfirmation(event.target.value)}
                  placeholder="Digite VERIFY TESTNET OCO"
                  autoComplete="off"
                  className="w-full rounded border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm"
                />
                <Button
                  variant="default"
                  size="sm"
                  onClick={verifyTestnetNativeOco}
                  disabled={liveActionLoading || testnetConfirmation !== "VERIFY TESTNET OCO" || !settings?.trading?.symbols[0]}
                >
                  {liveActionLoading ? "Verificando Testnet..." : "Executar prova controlada de OCO na Testnet"}
                </Button>
              </div>
            )}

            {liveTrading.reconciliation.issues.length > 0 && (
              <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-3">
                <p className="text-xs font-medium uppercase tracking-wide text-red-300">Divergências de proteção ou inventário</p>
                <ul className="mt-2 space-y-1 text-sm text-red-200">
                  {liveTrading.reconciliation.issues.map((issue) => <li key={issue}>• {issue}</li>)}
                </ul>
              </div>
            )}

            {liveTrading.armed && liveTrading.armed_until && (
              <p className="rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-300">
                A sessão está armada até {new Date(liveTrading.armed_until).toLocaleString("pt-BR")}. Cada nova ordem ainda precisa passar pelos demais limites de risco. Bloqueie esta sessão quando terminar.
              </p>
            )}

            {liveTrading.blockers.length > 0 && (
              <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] p-3">
                <p className="text-xs font-medium uppercase tracking-wide text-[var(--color-text-muted)]">Bloqueios ativos</p>
                <ul className="mt-2 space-y-1 text-sm text-[var(--color-text-muted)]">
                  {liveTrading.blockers.map((blocker) => <li key={blocker}>• {blocker}</li>)}
                </ul>
              </div>
            )}

            {liveActionError && <p className="text-sm text-red-400">{liveActionError}</p>}

            {liveTrading.armed ? (
              <Button variant="danger" size="sm" onClick={disarmLiveTrading} disabled={liveActionLoading}>
                {liveActionLoading ? "Bloqueando..." : "Bloquear conta real agora"}
              </Button>
            ) : liveTrading.armable ? (
              <div className="space-y-3 rounded-lg border border-yellow-500/30 bg-yellow-500/5 p-4">
                <p className="text-sm text-yellow-200">Armar não inicia operações; apenas libera ordens que já passarem por todos os limites de risco.</p>
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                  <input
                    type="password"
                    value={liveArmCode}
                    onChange={(event) => setLiveArmCode(event.target.value)}
                    placeholder="Código de armamento"
                    autoComplete="off"
                    className="rounded border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm"
                  />
                  <input
                    inputMode="numeric"
                    value={liveTotpCode}
                    onChange={(event) => setLiveTotpCode(event.target.value.replace(/\D/g, "").slice(0, 6))}
                    placeholder="Código TOTP"
                    autoComplete="one-time-code"
                    className="rounded border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm"
                  />
                  <input
                    value={liveConfirmation}
                    onChange={(event) => setLiveConfirmation(event.target.value)}
                    placeholder="Digite ARM LIVE TRADING"
                    autoComplete="off"
                    className="rounded border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm"
                  />
                </div>
                <Button
                  variant="danger"
                  size="sm"
                  onClick={armLiveTrading}
                  disabled={liveActionLoading || !liveArmCode || liveTotpCode.length !== 6 || liveConfirmation !== "ARM LIVE TRADING"}
                >
                  {liveActionLoading ? "Armando..." : "Armar execução em conta real"}
                </Button>
              </div>
            ) : null}
          </div>
        )}
      </Card>

      {/* Trading Configuration */}
      <Card>
        <CardHeader>
          <CardTitle>Configuração de Trading</CardTitle>
        </CardHeader>

        <div className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="mb-1 block text-xs text-[var(--color-text-muted)]">
                Modo de Trading
              </label>
              <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2">
                <Badge variant={tradingMode.variant}>
                  {tradingMode.label}
                </Badge>
                <span className="ml-2 text-sm text-[var(--color-text-muted)]">
                  {tradingMode.description}
                </span>
              </div>
            </div>

            <div>
              <label className="mb-1 block text-xs text-[var(--color-text-muted)]">
                Pares
              </label>
              <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm">
                {settings?.trading?.symbols.join(", ") ?? "BTCUSDT, ETHUSDT"}
              </div>
            </div>
          </div>
        </div>
      </Card>

      {/* Effective Risk Parameters */}
      <Card>
        <CardHeader>
          <CardTitle>Limites de Risco Efetivos</CardTitle>
          <p className="text-xs text-[var(--color-text-muted)]">
            Estes são os limites realmente usados pelo motor. Para alterá-los, atualize as variáveis
            TRADING_* no deploy e publique uma nova versão antes de armar a conta real.
          </p>
        </CardHeader>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {riskFields.map(({ key, label, min, max, format }) => (
            <div key={key} className="rounded-lg bg-[var(--color-background)] p-3">
              <label className="text-xs text-[var(--color-text-muted)] block mb-1.5">
                {label}
              </label>
              <p className="rounded border border-[var(--color-border)] bg-[var(--color-surface)] px-2 py-1.5 text-sm font-mono">
                {format === "pct" ? `${(risk[key] * 100).toFixed(2)}%` : risk[key].toFixed(2)}
              </p>
              <p className="mt-1 text-[10px] text-[var(--color-text-muted)]">
                {format === "pct"
                  ? `${(min * 100).toFixed(0)}% - ${(max * 100).toFixed(0)}%`
                  : `${min} - ${max}`}
              </p>
            </div>
          ))}
        </div>
      </Card>

      {/* API Documentation */}
      <Card>
        <CardHeader>
          <CardTitle>Documentação da API</CardTitle>
        </CardHeader>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <a
            href="/api/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-3 rounded-lg bg-[var(--color-background)] p-4 hover:bg-[var(--color-surface-hover)] transition-colors"
          >
            <ExternalLink className="h-5 w-5 text-[var(--color-primary)]" />
            <div>
              <p className="text-sm font-medium">Swagger UI</p>
              <p className="text-xs text-[var(--color-text-muted)]">Documentação interativa da API</p>
            </div>
          </a>
          <a
            href="/api/redoc"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-3 rounded-lg bg-[var(--color-background)] p-4 hover:bg-[var(--color-surface-hover)] transition-colors"
          >
            <ExternalLink className="h-5 w-5 text-[var(--color-primary)]" />
            <div>
              <p className="text-sm font-medium">ReDoc</p>
              <p className="text-xs text-[var(--color-text-muted)]">Documentação alternativa da API</p>
            </div>
          </a>
        </div>
      </Card>
    </div>
  );
}
