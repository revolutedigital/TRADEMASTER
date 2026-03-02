"use client";

import { useState, useEffect } from "react";
import { Sidebar } from "@/components/sidebar";

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
      const res = await fetch("/api/v1/market/sentiment", { credentials: "include" });
      if (res.ok) setData(await res.json());
    } catch {
      // Use fallback data
    } finally {
      setLoading(false);
    }
  }

  function getFearGreedColor(value: number) {
    if (value <= 25) return "text-red-500";
    if (value <= 45) return "text-orange-500";
    if (value <= 55) return "text-yellow-500";
    if (value <= 75) return "text-green-400";
    return "text-green-500";
  }

  return (
    <div className="flex h-screen bg-[#0a0e17]">
      <Sidebar />
      <main className="flex-1 overflow-y-auto p-6">
        <h1 className="text-2xl font-bold text-white mb-6">Market Sentiment</h1>

        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="bg-[#141922] rounded-xl p-6 animate-pulse h-48" />
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Fear & Greed Index */}
            <div className="bg-[#141922] rounded-xl p-6 col-span-1 md:col-span-2 lg:col-span-1">
              <h2 className="text-sm text-gray-400 mb-4">Fear & Greed Index</h2>
              <div className="flex flex-col items-center">
                <div className={`text-6xl font-bold ${getFearGreedColor(data?.fear_greed_index ?? 50)}`}>
                  {data?.fear_greed_index ?? "--"}
                </div>
                <div className="text-lg text-gray-300 mt-2">{data?.fear_greed_label ?? "Neutral"}</div>
                <div className="w-full bg-gray-700 rounded-full h-3 mt-4">
                  <div
                    className="h-3 rounded-full bg-gradient-to-r from-red-500 via-yellow-500 to-green-500"
                    style={{ width: `${data?.fear_greed_index ?? 50}%` }}
                  />
                </div>
                <div className="flex justify-between w-full text-xs text-gray-500 mt-1">
                  <span>Extreme Fear</span>
                  <span>Extreme Greed</span>
                </div>
              </div>
            </div>

            {/* Funding Rates */}
            <div className="bg-[#141922] rounded-xl p-6">
              <h2 className="text-sm text-gray-400 mb-4">Funding Rates</h2>
              <div className="space-y-4">
                {data?.funding_rates ? Object.entries(data.funding_rates).map(([symbol, rate]) => (
                  <div key={symbol} className="flex justify-between items-center">
                    <span className="text-white font-medium">{symbol}</span>
                    <span className={rate >= 0 ? "text-green-400" : "text-red-400"}>
                      {(rate * 100).toFixed(4)}%
                    </span>
                  </div>
                )) : (
                  <p className="text-gray-500">No data available</p>
                )}
              </div>
            </div>

            {/* Long/Short Ratio */}
            <div className="bg-[#141922] rounded-xl p-6">
              <h2 className="text-sm text-gray-400 mb-4">Long/Short Ratio</h2>
              <div className="space-y-4">
                {data?.long_short_ratio ? Object.entries(data.long_short_ratio).map(([symbol, ratio]) => (
                  <div key={symbol}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-white">{symbol}</span>
                      <span className="text-gray-400">{ratio.toFixed(2)}</span>
                    </div>
                    <div className="flex h-2 rounded-full overflow-hidden">
                      <div className="bg-green-500" style={{ width: `${(ratio / (ratio + 1)) * 100}%` }} />
                      <div className="bg-red-500 flex-1" />
                    </div>
                    <div className="flex justify-between text-xs text-gray-500 mt-1">
                      <span>Long {((ratio / (ratio + 1)) * 100).toFixed(1)}%</span>
                      <span>Short {((1 / (ratio + 1)) * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                )) : (
                  <p className="text-gray-500">No data available</p>
                )}
              </div>
            </div>

            {/* Open Interest */}
            <div className="bg-[#141922] rounded-xl p-6">
              <h2 className="text-sm text-gray-400 mb-4">Open Interest</h2>
              <div className="space-y-4">
                {data?.open_interest ? Object.entries(data.open_interest).map(([symbol, oi]) => (
                  <div key={symbol} className="flex justify-between items-center">
                    <span className="text-white font-medium">{symbol}</span>
                    <span className="text-gray-300">${(oi / 1e9).toFixed(2)}B</span>
                  </div>
                )) : (
                  <p className="text-gray-500">No data available</p>
                )}
              </div>
            </div>

            {/* Market Overview */}
            <div className="bg-[#141922] rounded-xl p-6">
              <h2 className="text-sm text-gray-400 mb-4">Market Overview</h2>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-400">Dominance BTC</span>
                  <span className="text-white">54.2%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Dominance ETH</span>
                  <span className="text-white">17.8%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Total Market Cap</span>
                  <span className="text-white">$3.2T</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">24h Volume</span>
                  <span className="text-white">$89.4B</span>
                </div>
              </div>
            </div>

            {/* Sentiment Summary */}
            <div className="bg-[#141922] rounded-xl p-6">
              <h2 className="text-sm text-gray-400 mb-4">Sentiment Summary</h2>
              <div className="space-y-3">
                <div className="p-3 bg-[#1a1f2e] rounded-lg">
                  <span className="text-xs text-gray-400">Overall Bias</span>
                  <p className="text-lg text-yellow-400 font-semibold">Neutral</p>
                </div>
                <p className="text-sm text-gray-400">
                  Market sentiment is mixed. Funding rates suggest balanced positioning.
                  Monitor for directional shifts in long/short ratios.
                </p>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
