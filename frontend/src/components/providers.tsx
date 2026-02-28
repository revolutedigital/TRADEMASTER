"use client";

import { useEffect, useRef } from "react";
import { usePortfolioStore } from "@/stores/portfolioStore";
import { useBinanceStream } from "@/hooks/useBinanceStream";

export function Providers({ children }: { children: React.ReactNode }) {
  const fetchSummary = usePortfolioStore((s) => s.fetchSummary);
  const fetchPositions = usePortfolioStore((s) => s.fetchPositions);
  const fetchRiskStatus = usePortfolioStore((s) => s.fetchRiskStatus);
  const fetchSignals = usePortfolioStore((s) => s.fetchSignals);
  const started = useRef(false);

  // Real-time prices + klines from Binance WebSocket (browser → Binance, no geo-restriction)
  useBinanceStream();

  useEffect(() => {
    if (started.current) return;
    started.current = true;

    // Initial portfolio data from backend
    fetchSummary();
    fetchPositions();
    fetchRiskStatus();
    fetchSignals();

    // Portfolio polling every 1s (fast trading mode)
    const portfolioInterval = setInterval(() => {
      fetchSummary();
      fetchPositions();
      fetchRiskStatus();
      fetchSignals();
    }, 1000);

    return () => {
      clearInterval(portfolioInterval);
    };
  }, [fetchSummary, fetchPositions, fetchRiskStatus, fetchSignals]);

  return <>{children}</>;
}
