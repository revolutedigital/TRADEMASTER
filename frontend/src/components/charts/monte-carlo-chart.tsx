"use client";

import { useRef, useEffect } from "react";

interface MonteCarloChartProps {
  paths: number[][];
  median: number;
  worst5pct: number;
  best5pct: number;
  initialValue: number;
  horizonDays: number;
}

export function MonteCarloChart({
  paths,
  median,
  worst5pct,
  best5pct,
  initialValue,
  horizonDays,
}: MonteCarloChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || paths.length === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const padding = { top: 30, right: 20, bottom: 40, left: 70 };

    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // Find min/max across all paths
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (const path of paths) {
      for (const val of path) {
        minVal = Math.min(minVal, val);
        maxVal = Math.max(maxVal, val);
      }
    }
    const range = maxVal - minVal || 1;

    const xScale = (i: number, total: number) => padding.left + (i / (total - 1)) * chartWidth;
    const yScale = (val: number) => padding.top + chartHeight - ((val - minVal) / range) * chartHeight;

    // Clear
    ctx.fillStyle = "#0a0e17";
    ctx.fillRect(0, 0, width, height);

    // Grid
    ctx.strokeStyle = "#1a1f2e";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const y = padding.top + (i / 5) * chartHeight;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();

      const val = maxVal - (i / 5) * range;
      ctx.fillStyle = "#6b7280";
      ctx.font = "11px monospace";
      ctx.textAlign = "right";
      ctx.fillText(`$${val.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, padding.left - 8, y + 4);
    }

    // X-axis labels
    ctx.fillStyle = "#6b7280";
    ctx.textAlign = "center";
    for (let i = 0; i <= 4; i++) {
      const day = Math.round((i / 4) * horizonDays);
      const x = xScale(day, horizonDays);
      ctx.fillText(`Day ${day}`, x, height - 10);
    }

    // Draw simulation paths (fan chart)
    const pathLen = paths[0]?.length || 0;
    for (const path of paths) {
      ctx.beginPath();
      ctx.strokeStyle = "rgba(99, 102, 241, 0.08)";
      ctx.lineWidth = 0.5;
      for (let i = 0; i < path.length; i++) {
        const x = xScale(i, pathLen);
        const y = yScale(path[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // Draw confidence bands
    const drawLine = (value: number, color: string, label: string, dash: number[] = []) => {
      const y = yScale(value);
      ctx.setLineDash(dash);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = color;
      ctx.font = "11px monospace";
      ctx.textAlign = "left";
      ctx.fillText(`${label}: $${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
        width - padding.right - 180, y - 6);
    };

    drawLine(best5pct, "#10b981", "95th pctl", [5, 3]);
    drawLine(median, "#f59e0b", "Median", []);
    drawLine(worst5pct, "#ef4444", "5th pctl", [5, 3]);
    drawLine(initialValue, "#6b7280", "Initial", [3, 3]);

    // Title
    ctx.fillStyle = "#e5e7eb";
    ctx.font = "bold 13px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(`Monte Carlo Simulation (${paths.length} paths, ${horizonDays} days)`, padding.left, 18);
  }, [paths, median, worst5pct, best5pct, initialValue, horizonDays]);

  return (
    <div className="bg-[#1a1f2e] rounded-xl p-4 border border-gray-800">
      <canvas
        ref={canvasRef}
        width={800}
        height={400}
        className="w-full h-auto"
      />
      <div className="flex justify-between mt-3 text-xs text-gray-400">
        <span>Worst 5%: ${worst5pct.toLocaleString()}</span>
        <span>Median: ${median.toLocaleString()}</span>
        <span>Best 5%: ${best5pct.toLocaleString()}</span>
      </div>
    </div>
  );
}
