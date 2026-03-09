"use client";

import { useState, useEffect, useCallback } from "react";
import { Bell, Plus, Trash2 } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { PageHeader } from "@/components/ui/page-header";
import { EmptyState } from "@/components/ui/empty-state";
import { Spinner } from "@/components/ui/progress";

interface PriceAlert {
  id: string;
  symbol: string;
  condition: "above" | "below";
  target_price: number;
  is_triggered: boolean;
  triggered_at: string | null;
  created_at: string;
}

export default function AlertsPage() {
  const [alerts, setAlerts] = useState<PriceAlert[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState<{ symbol: string; condition: "above" | "below"; target_price: string }>({ symbol: "BTCUSDT", condition: "above", target_price: "" });

  const fetchAlerts = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/alerts", { credentials: "include" });
      if (res.ok) setAlerts(await res.json());
    } catch {} finally { setLoading(false); }
  }, []);

  useEffect(() => { fetchAlerts(); }, [fetchAlerts]);

  async function createAlert(e: React.FormEvent) {
    e.preventDefault();
    try {
      const res = await fetch("/api/v1/alerts", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...form, target_price: parseFloat(form.target_price) }),
      });
      if (res.ok) { setShowForm(false); setForm({ symbol: "BTCUSDT", condition: "above", target_price: "" }); fetchAlerts(); }
    } catch {}
  }

  async function deleteAlert(id: string) {
    try {
      await fetch(`/api/v1/alerts/${id}`, { method: "DELETE", credentials: "include" });
      fetchAlerts();
    } catch {}
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Price Alerts"
        description="Get notified when prices hit your targets"
        actions={
          <Button variant={showForm ? "ghost" : "primary"} size="sm" onClick={() => setShowForm(!showForm)}>
            {showForm ? "Cancel" : <><Plus className="mr-1.5 h-4 w-4" /> New Alert</>}
          </Button>
        }
      />

      {showForm && (
        <Card>
          <CardContent>
            <form onSubmit={createAlert} className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
              <div>
                <label className="block text-xs font-medium uppercase tracking-wider text-[var(--color-text-muted)] mb-1.5">Symbol</label>
                <select
                  value={form.symbol}
                  onChange={(e) => setForm({ ...form, symbol: e.target.value })}
                  className="w-full rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm text-[var(--color-text)] focus:border-[var(--color-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20"
                >
                  <option value="BTCUSDT">BTCUSDT</option>
                  <option value="ETHUSDT">ETHUSDT</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-medium uppercase tracking-wider text-[var(--color-text-muted)] mb-1.5">Condition</label>
                <select
                  value={form.condition}
                  onChange={(e) => setForm({ ...form, condition: e.target.value as "above" | "below" })}
                  className="w-full rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm text-[var(--color-text)] focus:border-[var(--color-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20"
                >
                  <option value="above">Price Above</option>
                  <option value="below">Price Below</option>
                </select>
              </div>
              <Input
                label="Target Price"
                type="number"
                step="0.01"
                placeholder="e.g. 100000"
                value={form.target_price}
                onChange={(e) => setForm({ ...form, target_price: e.target.value })}
                required
              />
              <Button type="submit" variant="success" className="w-full">Create Alert</Button>
            </form>
          </CardContent>
        </Card>
      )}

      {loading ? (
        <div className="flex justify-center py-12"><Spinner size="lg" /></div>
      ) : alerts.length === 0 ? (
        <Card>
          <EmptyState
            icon={<Bell className="h-7 w-7" />}
            title="No price alerts configured"
            description="Create your first alert to get notified when prices hit your targets."
            action={{ label: "+ New Alert", onClick: () => setShowForm(true) }}
          />
        </Card>
      ) : (
        <div className="space-y-3">
          {alerts.map((alert) => (
            <Card key={alert.id} className={alert.is_triggered ? "border-[var(--color-warning)]/30" : ""}>
              <CardContent>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className={`h-3 w-3 rounded-full ${alert.is_triggered ? "bg-[var(--color-warning)]" : "bg-[var(--color-success)]"}`} />
                    <div>
                      <span className="font-semibold text-[var(--color-text)]">{alert.symbol}</span>
                      <span className="ml-2 text-[var(--color-text-muted)]">
                        {alert.condition === "above" ? ">" : "<"} ${alert.target_price.toLocaleString()}
                      </span>
                    </div>
                    {alert.is_triggered && (
                      <span className="text-xs rounded-full bg-[var(--color-warning)]/15 px-2.5 py-1 font-medium text-[var(--color-warning)]">
                        Triggered
                      </span>
                    )}
                  </div>
                  <Button variant="ghost" size="sm" onClick={() => deleteAlert(alert.id)} className="text-[var(--color-danger)] hover:text-[var(--color-danger)]">
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
