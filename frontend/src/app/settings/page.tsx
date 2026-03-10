"use client";

import { useState, useEffect } from "react";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { apiFetch } from "@/lib/utils";
import type { SystemHealth } from "@/lib/types";
import { Settings, Server, Database, Wifi, Save, ExternalLink } from "lucide-react";

interface RiskConfig {
  max_daily_drawdown: number;
  max_weekly_drawdown: number;
  max_monthly_drawdown: number;
  max_total_drawdown: number;
  atr_stop_multiplier: number;
  trailing_stop_activation: number;
  kelly_fraction: number;
  max_single_asset: number;
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

export default function SettingsPage() {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [settings, setSettingsData] = useState<FullSettings | null>(null);
  const [risk, setRisk] = useState<RiskConfig>({
    max_daily_drawdown: 0.03,
    max_weekly_drawdown: 0.07,
    max_monthly_drawdown: 0.10,
    max_total_drawdown: 0.15,
    atr_stop_multiplier: 2.0,
    trailing_stop_activation: 0.015,
    kelly_fraction: 0.15,
    max_single_asset: 0.30,
  });
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    apiFetch<SystemHealth>("/api/v1/system/health")
      .then(setHealth)
      .catch(() => {});
    apiFetch<FullSettings>("/api/v1/settings/")
      .then((s) => {
        setSettingsData(s);
        setRisk(s.risk);
      })
      .catch(() => {});
  }, []);

  const saveRiskSettings = async () => {
    setSaving(true);
    setSaved(false);
    try {
      const updated = await apiFetch<RiskConfig>("/api/v1/settings/risk", {
        method: "PUT",
        body: JSON.stringify(risk),
      });
      setRisk(updated);
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } catch (err) {
      console.error("Failed to save settings:", err);
    }
    setSaving(false);
  };

  const riskFields: { key: keyof RiskConfig; label: string; min: number; max: number; step: number; format: "pct" | "num" }[] = [
    { key: "max_daily_drawdown", label: "Drawdown Máximo Diário", min: 0.01, max: 0.20, step: 0.01, format: "pct" },
    { key: "max_weekly_drawdown", label: "Drawdown Máximo Semanal", min: 0.02, max: 0.30, step: 0.01, format: "pct" },
    { key: "max_monthly_drawdown", label: "Drawdown Máximo Mensal", min: 0.03, max: 0.50, step: 0.01, format: "pct" },
    { key: "max_total_drawdown", label: "Drawdown Máximo Total", min: 0.05, max: 0.50, step: 0.01, format: "pct" },
    { key: "atr_stop_multiplier", label: "Multiplicador ATR Stop", min: 0.5, max: 5.0, step: 0.1, format: "num" },
    { key: "trailing_stop_activation", label: "Ativação Trailing Stop", min: 0.005, max: 0.10, step: 0.005, format: "pct" },
    { key: "kelly_fraction", label: "Fração de Kelly", min: 0.05, max: 0.50, step: 0.05, format: "pct" },
    { key: "max_single_asset", label: "Máximo por Ativo", min: 0.10, max: 1.0, step: 0.05, format: "pct" },
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
                <Badge variant="warning">
                  {settings?.trading.trading_mode === "testnet" ? "Testnet" : "Live"}
                </Badge>
                <span className="ml-2 text-sm text-[var(--color-text-muted)]">
                  Paper trading na Binance Testnet
                </span>
              </div>
            </div>

            <div>
              <label className="mb-1 block text-xs text-[var(--color-text-muted)]">
                Pares
              </label>
              <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm">
                {settings?.trading.symbols.join(", ") ?? "BTCUSDT, ETHUSDT"}
              </div>
            </div>
          </div>
        </div>
      </Card>

      {/* Risk Parameters - Editable */}
      <Card>
        <CardHeader>
          <CardTitle>Parâmetros de Risco</CardTitle>
          <div className="flex items-center gap-2">
            {saved && (
              <span className="text-xs text-green-400">Salvo!</span>
            )}
            <Button
              variant="primary"
              size="sm"
              onClick={saveRiskSettings}
              disabled={saving}
            >
              <Save className="mr-1.5 h-3.5 w-3.5" />
              {saving ? "Salvando..." : "Salvar Alterações"}
            </Button>
          </div>
        </CardHeader>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {riskFields.map(({ key, label, min, max, step, format }) => (
            <div key={key} className="rounded-lg bg-[var(--color-background)] p-3">
              <label className="text-xs text-[var(--color-text-muted)] block mb-1.5">
                {label}
              </label>
              <input
                type="number"
                value={risk[key]}
                onChange={(e) => setRisk({ ...risk, [key]: Number(e.target.value) })}
                min={min}
                max={max}
                step={step}
                className="w-full rounded border border-[var(--color-border)] bg-[var(--color-surface)] px-2 py-1.5 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
              />
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
