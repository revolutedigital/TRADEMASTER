"use client";

import { useState, useEffect } from "react";
import { Sidebar } from "@/components/ui/sidebar";

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
      const res = await fetch(`/api/v1/portfolio/fees?period=${period}`, { credentials: "include" });
      if (res.ok) setData(await res.json());
    } catch {} finally { setLoading(false); }
  }

  return (
    <div className="flex h-screen bg-[#0a0e17]">
      <Sidebar />
      <main className="flex-1 overflow-y-auto p-6">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-bold text-white">Fee Analysis</h1>
          <div className="flex gap-2">
            {["7d", "30d", "90d", "1y"].map((p) => (
              <button key={p} onClick={() => setPeriod(p)} className={`px-3 py-1 rounded-lg text-sm ${period === p ? "bg-blue-600 text-white" : "bg-[#141922] text-gray-400 hover:text-white"}`}>
                {p}
              </button>
            ))}
          </div>
        </div>

        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[...Array(3)].map((_, i) => <div key={i} className="bg-[#141922] rounded-xl p-6 animate-pulse h-32" />)}
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
              <div className="bg-[#141922] rounded-xl p-6">
                <h2 className="text-sm text-gray-400">Total Fees Paid</h2>
                <p className="text-3xl font-bold text-white mt-2">${data?.total_fees.toFixed(2) ?? "0.00"}</p>
              </div>
              <div className="bg-[#141922] rounded-xl p-6">
                <h2 className="text-sm text-gray-400">Fee Impact on Returns</h2>
                <p className="text-3xl font-bold text-red-400 mt-2">{data?.fee_impact_pct.toFixed(2) ?? "0.00"}%</p>
              </div>
              <div className="bg-[#141922] rounded-xl p-6">
                <h2 className="text-sm text-gray-400">Avg Fee per Trade</h2>
                <p className="text-3xl font-bold text-yellow-400 mt-2">${((data?.total_fees ?? 0) / Math.max(Object.keys(data?.fees_by_symbol ?? {}).length, 1)).toFixed(2)}</p>
              </div>
            </div>

            <div className="bg-[#141922] rounded-xl p-6">
              <h2 className="text-sm text-gray-400 mb-4">Fees by Symbol</h2>
              <div className="space-y-3">
                {data?.fees_by_symbol && Object.entries(data.fees_by_symbol).sort(([, a], [, b]) => b - a).map(([symbol, fee]) => (
                  <div key={symbol} className="flex items-center gap-4">
                    <span className="text-white font-medium w-24">{symbol}</span>
                    <div className="flex-1 bg-gray-700 rounded-full h-2">
                      <div className="bg-orange-500 h-2 rounded-full" style={{ width: `${(fee / (data.total_fees || 1)) * 100}%` }} />
                    </div>
                    <span className="text-gray-300 w-24 text-right">${fee.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
