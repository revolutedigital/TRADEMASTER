import { create } from "zustand";
import type { Kline, TickerPrice, TimeInterval } from "@/lib/types";

const BINANCE_REST = "https://api.binance.com/api/v3";

interface MarketState {
  // Data
  prices: Record<string, TickerPrice>;
  klines: Record<string, Kline[]>;
  selectedSymbol: string;
  selectedInterval: TimeInterval;

  // Actions
  setSelectedSymbol: (symbol: string) => void;
  setSelectedInterval: (interval: TimeInterval) => void;
  updatePrice: (symbol: string, ticker: TickerPrice) => void;
  appendKline: (key: string, kline: Kline) => void;
  fetchKlines: (symbol: string, interval: TimeInterval) => Promise<void>;
}

export const useMarketStore = create<MarketState>((set) => ({
  prices: {},
  klines: {},
  selectedSymbol: "BTCUSDT",
  selectedInterval: "1h",

  setSelectedSymbol: (symbol) => set({ selectedSymbol: symbol }),
  setSelectedInterval: (interval) => set({ selectedInterval: interval }),

  updatePrice: (symbol, ticker) =>
    set((state) => ({
      prices: { ...state.prices, [symbol]: ticker },
    })),

  appendKline: (key, kline) =>
    set((state) => {
      const existing = state.klines[key] || [];
      const lastKline = existing[existing.length - 1];

      // Update last candle if same timestamp, otherwise append
      if (lastKline && lastKline.open_time === kline.open_time) {
        return {
          klines: {
            ...state.klines,
            [key]: [...existing.slice(0, -1), kline],
          },
        };
      }
      return {
        klines: {
          ...state.klines,
          [key]: [...existing.slice(-499), kline],
        },
      };
    }),

  // Fetch historical klines directly from Binance public REST API (no backend needed)
  fetchKlines: async (symbol, interval) => {
    try {
      const res = await fetch(
        `${BINANCE_REST}/klines?symbol=${symbol}&interval=${interval}&limit=500`
      );
      if (!res.ok) throw new Error(`Binance API ${res.status}`);

      // Binance returns: [[openTime, open, high, low, close, volume, closeTime, ...], ...]
      const raw: (string | number)[][] = await res.json();

      const data: Kline[] = raw.map((k) => ({
        open_time: Number(k[0]),
        open: parseFloat(String(k[1])),
        high: parseFloat(String(k[2])),
        low: parseFloat(String(k[3])),
        close: parseFloat(String(k[4])),
        volume: parseFloat(String(k[5])),
      }));

      set((state) => ({
        klines: { ...state.klines, [`${symbol}_${interval}`]: data },
      }));
    } catch (err) {
      console.error("Failed to fetch klines from Binance:", err);
    }
  },
}));
