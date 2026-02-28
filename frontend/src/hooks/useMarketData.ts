"use client";

import { useEffect } from "react";
import { useMarketStore } from "@/stores/marketStore";

/**
 * Hook for market data consumption.
 * Real-time prices + klines come from useBinanceStream (browser → Binance WS).
 * This hook only fetches initial historical klines from the backend
 * and exposes store selectors for components.
 */
export function useMarketData() {
  const {
    prices,
    klines,
    selectedSymbol,
    selectedInterval,
    setSelectedSymbol,
    setSelectedInterval,
    fetchKlines,
  } = useMarketStore();

  // Fetch historical klines when symbol or interval changes
  useEffect(() => {
    fetchKlines(selectedSymbol, selectedInterval);
  }, [selectedSymbol, selectedInterval, fetchKlines]);

  const currentPrice = prices[selectedSymbol];
  const currentKlines =
    klines[`${selectedSymbol}_${selectedInterval}`] || [];

  return {
    prices,
    currentPrice,
    currentKlines,
    selectedSymbol,
    selectedInterval,
    setSelectedSymbol,
    setSelectedInterval,
  };
}
