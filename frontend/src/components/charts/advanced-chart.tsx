"use client";

import { useState, useRef, useEffect, useCallback } from "react";

interface Candle {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface DrawingTool {
  id: string;
  type: "trendline" | "horizontal" | "fibonacci" | "rectangle";
  points: { x: number; y: number }[];
  color: string;
}

interface AdvancedChartProps {
  symbol: string;
  interval?: string;
}

export function AdvancedChart({ symbol, interval = "1h" }: AdvancedChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const [candles, setCandles] = useState<Candle[]>([]);
  const [drawings, setDrawings] = useState<DrawingTool[]>([]);
  const [activeTool, setActiveTool] = useState<string | null>(null);
  const [drawingPoints, setDrawingPoints] = useState<{ x: number; y: number }[]>([]);
  const [timeframe, setTimeframe] = useState(interval);
  const [indicators, setIndicators] = useState<string[]>(["sma_20"]);
  const [crosshair, setCrosshair] = useState<{ x: number; y: number } | null>(null);

  useEffect(() => {
    fetchCandles();
  }, [symbol, timeframe]);

  useEffect(() => {
    drawChart();
  }, [candles, drawings, indicators, crosshair]);

  async function fetchCandles() {
    try {
      const res = await fetch(`/api/v1/market/klines/${symbol}?interval=${timeframe}&limit=200`, { credentials: "include" });
      if (res.ok) {
        const data = await res.json();
        setCandles(data.map((c: Record<string, unknown>) => ({
          time: new Date(c.open_time as string).getTime(),
          open: Number(c.open),
          high: Number(c.high),
          low: Number(c.low),
          close: Number(c.close),
          volume: Number(c.volume),
        })));
      }
    } catch {}
  }

  function drawChart() {
    const canvas = canvasRef.current;
    if (!canvas || candles.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { width, height } = canvas;
    const chartHeight = height * 0.75;
    const volumeHeight = height * 0.2;
    const padding = { top: 10, right: 60, bottom: 30, left: 10 };

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#0a0e17";
    ctx.fillRect(0, 0, width, height);

    const prices = candles.flatMap((c) => [c.high, c.low]);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = maxPrice - minPrice || 1;
    const maxVolume = Math.max(...candles.map((c) => c.volume));

    const candleWidth = Math.max(2, (width - padding.left - padding.right) / candles.length - 1);

    // Draw price grid
    ctx.strokeStyle = "#1a1f2e";
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 5; i++) {
      const y = padding.top + (chartHeight - padding.top) * (i / 5);
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();

      const price = maxPrice - (priceRange * i) / 5;
      ctx.fillStyle = "#6b7280";
      ctx.font = "10px monospace";
      ctx.fillText(price.toFixed(2), width - padding.right + 5, y + 3);
    }

    // Draw candles
    candles.forEach((candle, i) => {
      const x = padding.left + i * (candleWidth + 1);
      const isGreen = candle.close >= candle.open;

      // Wick
      const wickX = x + candleWidth / 2;
      const highY = padding.top + ((maxPrice - candle.high) / priceRange) * (chartHeight - padding.top);
      const lowY = padding.top + ((maxPrice - candle.low) / priceRange) * (chartHeight - padding.top);
      ctx.beginPath();
      ctx.moveTo(wickX, highY);
      ctx.lineTo(wickX, lowY);
      ctx.strokeStyle = isGreen ? "#22c55e" : "#ef4444";
      ctx.lineWidth = 1;
      ctx.stroke();

      // Body
      const openY = padding.top + ((maxPrice - candle.open) / priceRange) * (chartHeight - padding.top);
      const closeY = padding.top + ((maxPrice - candle.close) / priceRange) * (chartHeight - padding.top);
      const bodyTop = Math.min(openY, closeY);
      const bodyHeight = Math.max(Math.abs(closeY - openY), 1);
      ctx.fillStyle = isGreen ? "#22c55e" : "#ef4444";
      ctx.fillRect(x, bodyTop, candleWidth, bodyHeight);

      // Volume
      const volHeight = (candle.volume / maxVolume) * volumeHeight;
      const volY = chartHeight + (volumeHeight - volHeight);
      ctx.fillStyle = isGreen ? "rgba(34, 197, 94, 0.3)" : "rgba(239, 68, 68, 0.3)";
      ctx.fillRect(x, volY, candleWidth, volHeight);
    });

    // Draw SMA indicator
    if (indicators.includes("sma_20") && candles.length >= 20) {
      ctx.beginPath();
      ctx.strokeStyle = "#f59e0b";
      ctx.lineWidth = 1.5;
      for (let i = 19; i < candles.length; i++) {
        const sum = candles.slice(i - 19, i + 1).reduce((s, c) => s + c.close, 0);
        const sma = sum / 20;
        const x = padding.left + i * (candleWidth + 1) + candleWidth / 2;
        const y = padding.top + ((maxPrice - sma) / priceRange) * (chartHeight - padding.top);
        if (i === 19) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // Draw drawings
    drawings.forEach((drawing) => {
      ctx.strokeStyle = drawing.color;
      ctx.lineWidth = 2;
      if (drawing.type === "horizontal" && drawing.points.length >= 1) {
        ctx.beginPath();
        ctx.setLineDash([5, 5]);
        ctx.moveTo(0, drawing.points[0].y);
        ctx.lineTo(width, drawing.points[0].y);
        ctx.stroke();
        ctx.setLineDash([]);
      } else if (drawing.type === "trendline" && drawing.points.length >= 2) {
        ctx.beginPath();
        ctx.moveTo(drawing.points[0].x, drawing.points[0].y);
        ctx.lineTo(drawing.points[1].x, drawing.points[1].y);
        ctx.stroke();
      }
    });

    // Draw crosshair
    if (crosshair) {
      ctx.strokeStyle = "rgba(255, 255, 255, 0.3)";
      ctx.lineWidth = 0.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(crosshair.x, 0);
      ctx.lineTo(crosshair.x, height);
      ctx.moveTo(0, crosshair.y);
      ctx.lineTo(width, crosshair.y);
      ctx.stroke();
      ctx.setLineDash([]);

      // Price label at crosshair
      const crossPrice = maxPrice - ((crosshair.y - padding.top) / (chartHeight - padding.top)) * priceRange;
      if (crossPrice > 0) {
        ctx.fillStyle = "#3b82f6";
        ctx.fillRect(width - padding.right, crosshair.y - 8, 55, 16);
        ctx.fillStyle = "#ffffff";
        ctx.font = "10px monospace";
        ctx.fillText(crossPrice.toFixed(2), width - padding.right + 3, crosshair.y + 3);
      }
    }
  }

  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!activeTool) return;
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const point = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    const newPoints = [...drawingPoints, point];
    setDrawingPoints(newPoints);

    const requiredPoints = activeTool === "horizontal" ? 1 : 2;
    if (newPoints.length >= requiredPoints) {
      setDrawings((prev) => [
        ...prev,
        { id: `d-${Date.now()}`, type: activeTool as DrawingTool["type"], points: newPoints, color: "#3b82f6" },
      ]);
      setDrawingPoints([]);
      setActiveTool(null);
    }
  }, [activeTool, drawingPoints]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    setCrosshair({ x: e.clientX - rect.left, y: e.clientY - rect.top });
  }, []);

  const tools = [
    { id: "trendline", label: "Trend Line", icon: "╱" },
    { id: "horizontal", label: "Horizontal", icon: "━" },
    { id: "fibonacci", label: "Fibonacci", icon: "F" },
  ];

  const timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"];

  return (
    <div className="bg-[#0a0e17] rounded-xl border border-gray-800">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-gray-800">
        <div className="flex items-center gap-2">
          <span className="text-white font-semibold">{symbol}</span>
          <div className="flex gap-1 ml-4">
            {timeframes.map((tf) => (
              <button key={tf} onClick={() => setTimeframe(tf)} className={`px-2 py-0.5 text-xs rounded ${timeframe === tf ? "bg-blue-600 text-white" : "text-gray-400 hover:text-white"}`}>
                {tf}
              </button>
            ))}
          </div>
        </div>
        <div className="flex items-center gap-2">
          {tools.map((tool) => (
            <button key={tool.id} onClick={() => setActiveTool(activeTool === tool.id ? null : tool.id)} title={tool.label} className={`w-7 h-7 flex items-center justify-center rounded text-sm ${activeTool === tool.id ? "bg-blue-600 text-white" : "text-gray-400 hover:text-white hover:bg-gray-700"}`}>
              {tool.icon}
            </button>
          ))}
          {drawings.length > 0 && (
            <button onClick={() => setDrawings([])} className="text-xs text-red-400 hover:text-red-300 ml-2">
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Chart */}
      <div className="relative">
        <canvas
          ref={canvasRef}
          width={900}
          height={500}
          className="w-full cursor-crosshair"
          onClick={handleCanvasClick}
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setCrosshair(null)}
        />
      </div>
    </div>
  );
}
