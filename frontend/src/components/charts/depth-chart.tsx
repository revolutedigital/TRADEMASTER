"use client";

import { useState, useEffect, useRef } from "react";

interface DepthLevel {
  price: number;
  quantity: number;
  total: number;
}

interface DepthChartProps {
  symbol: string;
}

export function DepthChart({ symbol }: DepthChartProps) {
  const [bids, setBids] = useState<DepthLevel[]>([]);
  const [asks, setAsks] = useState<DepthLevel[]>([]);
  const [loading, setLoading] = useState(true);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    fetchDepth();
    const interval = setInterval(fetchDepth, 5000);
    return () => clearInterval(interval);
  }, [symbol]);

  useEffect(() => {
    if (bids.length > 0 && asks.length > 0) drawChart();
  }, [bids, asks]);

  async function fetchDepth() {
    try {
      const res = await fetch(`/api/v1/market/depth/${symbol}`, { credentials: "include" });
      if (res.ok) {
        const data = await res.json();
        processBids(data.bids || []);
        processAsks(data.asks || []);
      }
    } catch {} finally { setLoading(false); }
  }

  function processBids(raw: [string, string][]) {
    let total = 0;
    const processed = raw.slice(0, 25).map(([price, qty]) => {
      total += parseFloat(qty);
      return { price: parseFloat(price), quantity: parseFloat(qty), total };
    });
    setBids(processed);
  }

  function processAsks(raw: [string, string][]) {
    let total = 0;
    const processed = raw.slice(0, 25).map(([price, qty]) => {
      total += parseFloat(qty);
      return { price: parseFloat(price), quantity: parseFloat(qty), total };
    });
    setAsks(processed);
  }

  function drawChart() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);

    const maxTotal = Math.max(
      bids.length > 0 ? bids[bids.length - 1].total : 0,
      asks.length > 0 ? asks[asks.length - 1].total : 0,
    );
    if (maxTotal === 0) return;

    const midX = width / 2;

    // Draw bids (green, left side)
    ctx.beginPath();
    ctx.moveTo(midX, height);
    bids.forEach((level, i) => {
      const x = midX - (i / bids.length) * midX;
      const y = height - (level.total / maxTotal) * height * 0.9;
      ctx.lineTo(x, y);
    });
    ctx.lineTo(0, height);
    ctx.closePath();
    ctx.fillStyle = "rgba(34, 197, 94, 0.2)";
    ctx.fill();
    ctx.strokeStyle = "rgba(34, 197, 94, 0.8)";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw asks (red, right side)
    ctx.beginPath();
    ctx.moveTo(midX, height);
    asks.forEach((level, i) => {
      const x = midX + (i / asks.length) * midX;
      const y = height - (level.total / maxTotal) * height * 0.9;
      ctx.lineTo(x, y);
    });
    ctx.lineTo(width, height);
    ctx.closePath();
    ctx.fillStyle = "rgba(239, 68, 68, 0.2)";
    ctx.fill();
    ctx.strokeStyle = "rgba(239, 68, 68, 0.8)";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Mid price line
    ctx.beginPath();
    ctx.setLineDash([4, 4]);
    ctx.moveTo(midX, 0);
    ctx.lineTo(midX, height);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.3)";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.setLineDash([]);
  }

  if (loading) return <div className="bg-[#141922] rounded-xl p-6 h-64 animate-pulse" />;

  return (
    <div className="bg-[#141922] rounded-xl p-4">
      <div className="flex justify-between items-center mb-3">
        <h3 className="text-sm text-gray-400">Order Book Depth</h3>
        <div className="flex gap-4 text-xs">
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-green-500" /> Bids</span>
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-red-500" /> Asks</span>
        </div>
      </div>
      <canvas ref={canvasRef} width={600} height={200} className="w-full h-48" />
      <div className="flex justify-between text-xs text-gray-500 mt-1">
        <span>{bids.length > 0 ? `$${bids[bids.length - 1]?.price.toLocaleString()}` : ""}</span>
        <span>Spread</span>
        <span>{asks.length > 0 ? `$${asks[asks.length - 1]?.price.toLocaleString()}` : ""}</span>
      </div>
    </div>
  );
}
