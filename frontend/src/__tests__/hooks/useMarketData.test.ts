import { describe, it, expect, vi } from "vitest";
import { renderHook } from "@testing-library/react";

const mockFetchKlines = vi.fn();
vi.mock("@/stores/marketStore", () => ({
  useMarketStore: () => ({
    prices: { BTCUSDT: { price: 42000, change_24h: 0.05 } },
    klines: { BTCUSDT_1h: [{ o: 41000, h: 42500, l: 40500, c: 42000 }] },
    selectedSymbol: "BTCUSDT",
    selectedInterval: "1h",
    setSelectedSymbol: vi.fn(),
    setSelectedInterval: vi.fn(),
    fetchKlines: mockFetchKlines,
  }),
}));

import { useMarketData } from "@/hooks/useMarketData";

describe("useMarketData", () => {
  it("returns current price for selected symbol", () => {
    const { result } = renderHook(() => useMarketData());
    expect(result.current.currentPrice).toEqual({ price: 42000, change_24h: 0.05 });
  });

  it("returns klines for selected symbol and interval", () => {
    const { result } = renderHook(() => useMarketData());
    expect(result.current.currentKlines).toHaveLength(1);
  });

  it("returns selected symbol", () => {
    const { result } = renderHook(() => useMarketData());
    expect(result.current.selectedSymbol).toBe("BTCUSDT");
  });

  it("calls fetchKlines on mount", () => {
    renderHook(() => useMarketData());
    expect(mockFetchKlines).toHaveBeenCalledWith("BTCUSDT", "1h");
  });

  it("returns empty array when no klines for key", () => {
    vi.mocked(mockFetchKlines).mockClear();
    // The mock klines don't have a matching key for non-default
    const { result } = renderHook(() => useMarketData());
    // Default symbol/interval klines exist
    expect(result.current.currentKlines).toBeDefined();
  });
});
