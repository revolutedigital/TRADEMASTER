"use client";

import { useState, useEffect } from "react";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { apiFetch } from "@/lib/utils";
import type { SystemHealth } from "@/lib/types";
import { Settings, Server, Database, Wifi } from "lucide-react";

export default function SettingsPage() {
  const [health, setHealth] = useState<SystemHealth | null>(null);

  useEffect(() => {
    apiFetch<SystemHealth>("/api/v1/system/health")
      .then(setHealth)
      .catch(() => {});
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Settings className="h-6 w-6 text-[var(--color-primary)]" />
        <h1 className="text-2xl font-bold">Settings</h1>
      </div>

      {/* System Health */}
      <Card>
        <CardHeader>
          <CardTitle>System Status</CardTitle>
          <Badge variant={health?.status === "healthy" ? "success" : "warning"}>
            {health?.status ?? "Loading..."}
          </Badge>
        </CardHeader>

        <div className="grid grid-cols-3 gap-4">
          <div className="flex items-center gap-3 rounded-lg bg-[var(--color-background)] p-4">
            <Server className="h-5 w-5 text-[var(--color-text-muted)]" />
            <div>
              <p className="text-xs text-[var(--color-text-muted)]">API Server</p>
              <p className="text-sm font-medium text-green-400">
                {health?.services?.api ?? "Unknown"}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3 rounded-lg bg-[var(--color-background)] p-4">
            <Database className="h-5 w-5 text-[var(--color-text-muted)]" />
            <div>
              <p className="text-xs text-[var(--color-text-muted)]">Database</p>
              <p className="text-sm font-medium">
                {health?.services?.database ?? "Unknown"}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3 rounded-lg bg-[var(--color-background)] p-4">
            <Wifi className="h-5 w-5 text-[var(--color-text-muted)]" />
            <div>
              <p className="text-xs text-[var(--color-text-muted)]">Binance WS</p>
              <p className="text-sm font-medium">
                {health?.services?.binance ?? "Unknown"}
              </p>
            </div>
          </div>
        </div>

        {health && (
          <p className="mt-3 text-xs text-[var(--color-text-muted)]">
            Version: {health.version} | Uptime: {Math.floor(health.uptime / 60)}m
          </p>
        )}
      </Card>

      {/* Trading Configuration */}
      <Card>
        <CardHeader>
          <CardTitle>Trading Configuration</CardTitle>
        </CardHeader>

        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="mb-1 block text-xs text-[var(--color-text-muted)]">
                Trading Mode
              </label>
              <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2">
                <Badge variant="warning">Testnet</Badge>
                <span className="ml-2 text-sm text-[var(--color-text-muted)]">
                  Paper trading on Binance Testnet
                </span>
              </div>
            </div>

            <div>
              <label className="mb-1 block text-xs text-[var(--color-text-muted)]">
                AI Trading
              </label>
              <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2">
                <Badge variant="success">Active</Badge>
                <span className="ml-2 text-sm text-[var(--color-text-muted)]">
                  Fully autonomous mode
                </span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div className="rounded-lg bg-[var(--color-background)] p-3">
              <p className="text-xs text-[var(--color-text-muted)]">Symbols</p>
              <p className="mt-1 text-sm font-medium">BTCUSDT, ETHUSDT</p>
            </div>
            <div className="rounded-lg bg-[var(--color-background)] p-3">
              <p className="text-xs text-[var(--color-text-muted)]">Max Risk per Trade</p>
              <p className="mt-1 text-sm font-medium">2%</p>
            </div>
            <div className="rounded-lg bg-[var(--color-background)] p-3">
              <p className="text-xs text-[var(--color-text-muted)]">Max Total Exposure</p>
              <p className="mt-1 text-sm font-medium">60%</p>
            </div>
          </div>
        </div>
      </Card>

      {/* Risk Parameters */}
      <Card>
        <CardHeader>
          <CardTitle>Risk Parameters</CardTitle>
        </CardHeader>

        <div className="grid grid-cols-4 gap-4">
          {[
            { label: "Max Daily Drawdown", value: "3%" },
            { label: "Max Weekly Drawdown", value: "7%" },
            { label: "Max Monthly Drawdown", value: "10%" },
            { label: "Max Total Drawdown", value: "15%" },
            { label: "ATR Stop Multiplier", value: "2.0x" },
            { label: "Trailing Stop Activation", value: "1.5%" },
            { label: "Kelly Fraction", value: "15%" },
            { label: "Max Single Asset", value: "30%" },
          ].map(({ label, value }) => (
            <div key={label} className="rounded-lg bg-[var(--color-background)] p-3">
              <p className="text-xs text-[var(--color-text-muted)]">{label}</p>
              <p className="mt-1 text-sm font-mono font-medium">{value}</p>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
