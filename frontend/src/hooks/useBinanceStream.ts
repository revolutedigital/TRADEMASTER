"use client";

import { useEffect, useRef, useCallback } from "react";
import { useMarketStore } from "@/stores/marketStore";
import type { TickerPrice, Kline } from "@/lib/types";

const BINANCE_WS = "wss://stream.binance.com:9443/stream";
const SYMBOLS = ["btcusdt", "ethusdt"];
const THROTTLE_MS = 100; // Update price at most every 100ms per symbol

/**
 * Connects directly to Binance public WebSocket for real-time trading experience.
 *
 * Streams:
 * - @aggTrade  → every trade execution (price ticking like Binance)
 * - @miniTicker → 24h stats (high, low, volume, change) every ~1s
 * - @kline      → candlestick updates for chart
 */
export function useBinanceStream() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout>>(undefined);
  const updatePrice = useMarketStore((s) => s.updatePrice);
  const appendKline = useMarketStore((s) => s.appendKline);
  const selectedInterval = useMarketStore((s) => s.selectedInterval);

  // Throttle: buffer latest trade price per symbol, flush at THROTTLE_MS
  const tradeBufRef = useRef<Record<string, number>>({});
  const statsRef = useRef<Record<string, Omit<TickerPrice, "price">>>({});
  const flushRef = useRef<ReturnType<typeof setInterval>>(undefined);

  const flush = useCallback(() => {
    const buf = tradeBufRef.current;
    const stats = statsRef.current;
    for (const symbol of Object.keys(buf)) {
      const price = buf[symbol];
      const st = stats[symbol] || {
        symbol,
        change_24h: 0,
        volume_24h: 0,
        high_24h: 0,
        low_24h: 0,
      };
      updatePrice(symbol, { ...st, symbol, price });
    }
    tradeBufRef.current = {};
  }, [updatePrice]);

  useEffect(() => {
    let disposed = false;

    function connect() {
      if (disposed) return;
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.close();
      }

      const streams = [
        ...SYMBOLS.map((s) => `${s}@aggTrade`),
        ...SYMBOLS.map((s) => `${s}@miniTicker`),
        ...SYMBOLS.map((s) => `${s}@kline_${selectedInterval}`),
      ];

      const url = `${BINANCE_WS}?streams=${streams.join("/")}`;

      try {
        const ws = new WebSocket(url);

        ws.onopen = () => {
          console.log("[Binance WS] Connected - real-time trading streams active");
          // Start flush interval
          flushRef.current = setInterval(flush, THROTTLE_MS);
        };

        ws.onmessage = (event) => {
          try {
            const msg = JSON.parse(event.data);
            const data = msg.data;
            if (!data) return;

            const eventType: string = data.e;

            if (eventType === "aggTrade") {
              // Aggregated trade: buffer price for throttled update
              const symbol = (data.s as string).toUpperCase();
              tradeBufRef.current[symbol] = parseFloat(data.p);
            } else if (eventType === "24hrMiniTicker") {
              // 24h stats: update stats cache (used when flushing trades)
              const symbol = (data.s as string).toUpperCase();
              const close = parseFloat(data.c);
              const open = parseFloat(data.o);
              statsRef.current[symbol] = {
                symbol,
                change_24h: open > 0 ? (close - open) / open : 0,
                volume_24h: parseFloat(data.v),
                high_24h: parseFloat(data.h),
                low_24h: parseFloat(data.l),
              };
              // Also push immediate price update from ticker
              tradeBufRef.current[symbol] = close;
            } else if (eventType === "kline") {
              const k = data.k;
              const symbol = (data.s as string).toUpperCase();
              const kline: Kline = {
                open_time: k.t,
                open: parseFloat(k.o),
                high: parseFloat(k.h),
                low: parseFloat(k.l),
                close: parseFloat(k.c),
                volume: parseFloat(k.v),
              };
              appendKline(`${symbol}_${k.i}`, kline);
            }
          } catch {
            // Ignore parse errors
          }
        };

        ws.onclose = () => {
          if (disposed) return;
          console.log("[Binance WS] Disconnected, reconnecting in 1s...");
          wsRef.current = null;
          clearInterval(flushRef.current);
          reconnectRef.current = setTimeout(connect, 1000);
        };

        ws.onerror = () => {
          ws.close();
        };

        wsRef.current = ws;
      } catch {
        if (!disposed) {
          reconnectRef.current = setTimeout(connect, 2000);
        }
      }
    }

    connect();

    return () => {
      disposed = true;
      clearTimeout(reconnectRef.current);
      clearInterval(flushRef.current);
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [appendKline, selectedInterval, flush]);
}
