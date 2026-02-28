"use client";

import { useEffect, useRef } from "react";
import {
  createChart,
  type IChartApi,
  ColorType,
  type Time,
} from "lightweight-charts";

interface EquityChartProps {
  data: number[];
  height?: number;
  color?: string;
}

export function EquityChart({ data, height = 200, color = "#6366f1" }: EquityChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!containerRef.current || !data.length) return;

    if (chartRef.current) {
      chartRef.current.remove();
    }

    const chart = createChart(containerRef.current, {
      height,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#8888a0",
        fontSize: 11,
      },
      grid: {
        vertLines: { visible: false },
        horzLines: { color: "rgba(42, 42, 58, 0.3)" },
      },
      rightPriceScale: { borderColor: "#2a2a3a" },
      timeScale: { borderColor: "#2a2a3a", visible: false },
      crosshair: {
        vertLine: { visible: false },
        horzLine: { color: "#6366f1", width: 1, style: 2 },
      },
    });

    const series = chart.addAreaSeries({
      lineColor: color,
      topColor: `${color}40`,
      bottomColor: `${color}05`,
      lineWidth: 2,
    });

    const chartData = data.map((value, i) => ({
      time: i as Time,
      value,
    }));

    series.setData(chartData);
    chart.timeScale().fitContent();
    chartRef.current = chart;

    const ro = new ResizeObserver((entries) => {
      chart.applyOptions({ width: entries[0].contentRect.width });
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, [data, height, color]);

  return <div ref={containerRef} className="w-full" />;
}
