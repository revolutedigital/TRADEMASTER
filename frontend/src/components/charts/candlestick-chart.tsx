"use client";

import { useEffect, useRef, useCallback } from "react";
import {
  createChart,
  type IChartApi,
  type ISeriesApi,
  type CandlestickData,
  type Time,
  ColorType,
} from "lightweight-charts";
import type { Kline } from "@/lib/types";

interface CandlestickChartProps {
  data: Kline[];
  width?: number;
  height?: number;
}

export function CandlestickChart({ data, height = 450 }: CandlestickChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  const lastSetDataLenRef = useRef(0);
  const lastTimeRef = useRef(0);

  const initChart = useCallback(() => {
    if (!containerRef.current) return;

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
        vertLines: { color: "rgba(42, 42, 58, 0.5)" },
        horzLines: { color: "rgba(42, 42, 58, 0.5)" },
      },
      crosshair: {
        vertLine: { color: "#6366f1", width: 1, style: 2 },
        horzLine: { color: "#6366f1", width: 1, style: 2 },
      },
      rightPriceScale: {
        borderColor: "#2a2a3a",
      },
      timeScale: {
        borderColor: "#2a2a3a",
        timeVisible: true,
        secondsVisible: false,
      },
    });

    const candleSeries = chart.addCandlestickSeries({
      upColor: "#22c55e",
      downColor: "#ef4444",
      borderDownColor: "#ef4444",
      borderUpColor: "#22c55e",
      wickDownColor: "#ef4444",
      wickUpColor: "#22c55e",
    });

    const volumeSeries = chart.addHistogramSeries({
      priceFormat: { type: "volume" },
      priceScaleId: "volume",
    });

    chart.priceScale("volume").applyOptions({
      scaleMargins: { top: 0.8, bottom: 0 },
    });

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;
    volumeSeriesRef.current = volumeSeries;
    lastSetDataLenRef.current = 0;
    lastTimeRef.current = 0;

    const ro = new ResizeObserver((entries) => {
      const { width: w } = entries[0].contentRect;
      chart.applyOptions({ width: w });
    });
    ro.observe(containerRef.current);

    return () => ro.disconnect();
  }, [height]);

  useEffect(() => {
    const cleanup = initChart();
    return () => cleanup?.();
  }, [initChart]);

  useEffect(() => {
    if (!candleSeriesRef.current || !volumeSeriesRef.current || !data.length) return;

    const last = data[data.length - 1];
    const lastTimeSec = Math.floor(last.open_time / 1000);

    // Determine if this is a bulk load or a real-time tick.
    // Bulk load: first time, or big jump in data length (symbol/interval change).
    const isBulkLoad =
      lastSetDataLenRef.current === 0 ||
      Math.abs(data.length - lastSetDataLenRef.current) > 5;

    if (isBulkLoad) {
      // Sort data by time to avoid Lightweight Charts ordering errors
      const sorted = [...data].sort((a, b) => a.open_time - b.open_time);

      const candles: CandlestickData[] = sorted.map((k) => ({
        time: Math.floor(k.open_time / 1000) as Time,
        open: k.open,
        high: k.high,
        low: k.low,
        close: k.close,
      }));

      const volumes = sorted.map((k) => ({
        time: Math.floor(k.open_time / 1000) as Time,
        value: k.volume,
        color: k.close >= k.open ? "rgba(34,197,94,0.3)" : "rgba(239,68,68,0.3)",
      }));

      candleSeriesRef.current.setData(candles);
      volumeSeriesRef.current.setData(volumes);
      chartRef.current?.timeScale().fitContent();
      lastSetDataLenRef.current = data.length;
      lastTimeRef.current = lastTimeSec;
    } else {
      // Real-time update: only update/append the last candle.
      // Guard: time must be >= last known time for update() to work.
      if (lastTimeSec >= lastTimeRef.current) {
        try {
          candleSeriesRef.current.update({
            time: lastTimeSec as Time,
            open: last.open,
            high: last.high,
            low: last.low,
            close: last.close,
          });
          volumeSeriesRef.current.update({
            time: lastTimeSec as Time,
            value: last.volume,
            color: last.close >= last.open ? "rgba(34,197,94,0.3)" : "rgba(239,68,68,0.3)",
          });
          lastTimeRef.current = lastTimeSec;
          lastSetDataLenRef.current = data.length;
        } catch {
          // Fallback: full reload if update fails
          const sorted = [...data].sort((a, b) => a.open_time - b.open_time);
          candleSeriesRef.current.setData(
            sorted.map((k) => ({
              time: Math.floor(k.open_time / 1000) as Time,
              open: k.open,
              high: k.high,
              low: k.low,
              close: k.close,
            }))
          );
          volumeSeriesRef.current.setData(
            sorted.map((k) => ({
              time: Math.floor(k.open_time / 1000) as Time,
              value: k.volume,
              color: k.close >= k.open ? "rgba(34,197,94,0.3)" : "rgba(239,68,68,0.3)",
            }))
          );
          lastSetDataLenRef.current = data.length;
          lastTimeRef.current = Math.floor(sorted[sorted.length - 1].open_time / 1000);
        }
      }
      // If lastTimeSec < lastTimeRef, silently skip (stale WS data)
    }
  }, [data]);

  return <div ref={containerRef} className="w-full" />;
}
