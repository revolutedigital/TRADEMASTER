"use client";

import { useState, useEffect, useCallback } from "react";
import { Sidebar } from "@/components/ui/sidebar";

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
  const [form, setForm] = useState({ symbol: "BTCUSDT", condition: "above" as const, target_price: "" });

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
    <div className="flex h-screen bg-[#0a0e17]">
      <Sidebar />
      <main className="flex-1 overflow-y-auto p-6">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-bold text-white">Price Alerts</h1>
          <button onClick={() => setShowForm(!showForm)} className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors">
            {showForm ? "Cancel" : "+ New Alert"}
          </button>
        </div>

        {showForm && (
          <form onSubmit={createAlert} className="bg-[#141922] rounded-xl p-6 mb-6">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <select value={form.symbol} onChange={(e) => setForm({ ...form, symbol: e.target.value })} className="bg-[#1a1f2e] text-white rounded-lg px-3 py-2 border border-gray-700">
                <option value="BTCUSDT">BTCUSDT</option>
                <option value="ETHUSDT">ETHUSDT</option>
              </select>
              <select value={form.condition} onChange={(e) => setForm({ ...form, condition: e.target.value as "above" | "below" })} className="bg-[#1a1f2e] text-white rounded-lg px-3 py-2 border border-gray-700">
                <option value="above">Price Above</option>
                <option value="below">Price Below</option>
              </select>
              <input type="number" step="0.01" placeholder="Target Price" value={form.target_price} onChange={(e) => setForm({ ...form, target_price: e.target.value })} className="bg-[#1a1f2e] text-white rounded-lg px-3 py-2 border border-gray-700" required />
              <button type="submit" className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium transition-colors">Create</button>
            </div>
          </form>
        )}

        {loading ? (
          <div className="space-y-4">{[...Array(3)].map((_, i) => <div key={i} className="bg-[#141922] rounded-xl p-4 animate-pulse h-16" />)}</div>
        ) : alerts.length === 0 ? (
          <div className="bg-[#141922] rounded-xl p-12 text-center">
            <p className="text-gray-400 text-lg">No price alerts configured</p>
            <p className="text-gray-500 text-sm mt-2">Create your first alert to get notified when prices hit your targets.</p>
          </div>
        ) : (
          <div className="space-y-3">
            {alerts.map((alert) => (
              <div key={alert.id} className={`bg-[#141922] rounded-xl p-4 flex items-center justify-between ${alert.is_triggered ? "border border-yellow-500/30" : ""}`}>
                <div className="flex items-center gap-4">
                  <div className={`w-3 h-3 rounded-full ${alert.is_triggered ? "bg-yellow-500" : "bg-green-500"}`} />
                  <div>
                    <span className="text-white font-semibold">{alert.symbol}</span>
                    <span className="text-gray-400 ml-2">{alert.condition === "above" ? ">" : "<"} ${alert.target_price.toLocaleString()}</span>
                  </div>
                  {alert.is_triggered && <span className="text-xs bg-yellow-500/20 text-yellow-400 px-2 py-1 rounded">Triggered</span>}
                </div>
                <button onClick={() => deleteAlert(alert.id)} className="text-red-400 hover:text-red-300 text-sm">Delete</button>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
