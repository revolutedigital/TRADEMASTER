"use client";

import { usePortfolioStore } from "@/stores/portfolioStore";

/**
 * Hook for portfolio data consumption.
 * Portfolio data (positions, summary, risk, signals) is polled
 * every 5s by Providers. This hook simply exposes the store.
 */
export function usePortfolio() {
  return usePortfolioStore();
}
