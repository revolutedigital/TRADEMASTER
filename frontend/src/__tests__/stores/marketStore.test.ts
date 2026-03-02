import { describe, it, expect, vi, beforeEach } from "vitest";

const mockFetch = vi.fn();
global.fetch = mockFetch;

describe("marketStore", () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it("initializes with default values", async () => {
    const { useMarketStore } = await import("@/stores/marketStore");
    const state = useMarketStore.getState();
    expect(state.selectedSymbol).toBe("BTCUSDT");
    expect(state.selectedInterval).toBe("1h");
    expect(state.prices).toEqual({});
    expect(state.klines).toEqual({});
  });

  it("setSelectedSymbol updates symbol", async () => {
    const { useMarketStore } = await import("@/stores/marketStore");
    useMarketStore.getState().setSelectedSymbol("ETHUSDT");
    expect(useMarketStore.getState().selectedSymbol).toBe("ETHUSDT");
  });

  it("setSelectedInterval updates interval", async () => {
    const { useMarketStore } = await import("@/stores/marketStore");
    useMarketStore.getState().setSelectedInterval("15m");
    expect(useMarketStore.getState().selectedInterval).toBe("15m");
  });

  it("updatePrice sets price for symbol", async () => {
    const { useMarketStore } = await import("@/stores/marketStore");
    const ticker = {
      symbol: "BTCUSDT",
      price: 95000,
      change_24h: 0.02,
      volume_24h: 50000000,
      high_24h: 96000,
      low_24h: 93000,
    };
    useMarketStore.getState().updatePrice("BTCUSDT", ticker);
    expect(useMarketStore.getState().prices["BTCUSDT"]).toEqual(ticker);
  });

  it("appendKline adds new kline to list", async () => {
    const { useMarketStore } = await import("@/stores/marketStore");
    const kline = { open_time: 1000, open: 95000, high: 95500, low: 94500, close: 95200, volume: 100 };

    useMarketStore.getState().appendKline("BTCUSDT_1h", kline);

    const klines = useMarketStore.getState().klines["BTCUSDT_1h"];
    expect(klines).toBeDefined();
    expect(klines.length).toBe(1);
    expect(klines[0]).toEqual(kline);
  });

  it("appendKline updates last kline if same timestamp", async () => {
    const { useMarketStore } = await import("@/stores/marketStore");
    const kline1 = { open_time: 1000, open: 95000, high: 95500, low: 94500, close: 95200, volume: 100 };
    const kline2 = { open_time: 1000, open: 95000, high: 96000, low: 94500, close: 95800, volume: 200 };

    useMarketStore.getState().appendKline("BTCUSDT_1h", kline1);
    useMarketStore.getState().appendKline("BTCUSDT_1h", kline2);

    const klines = useMarketStore.getState().klines["BTCUSDT_1h"];
    expect(klines.length).toBe(1);
    expect(klines[0].close).toBe(95800);
    expect(klines[0].volume).toBe(200);
  });

  it("appendKline limits to 500 klines", async () => {
    const { useMarketStore } = await import("@/stores/marketStore");

    // Add 510 klines
    for (let i = 0; i < 510; i++) {
      useMarketStore.getState().appendKline("BTCUSDT_1h", {
        open_time: i * 3600000,
        open: 95000 + i,
        high: 96000 + i,
        low: 94000 + i,
        close: 95500 + i,
        volume: 100 + i,
      });
    }

    const klines = useMarketStore.getState().klines["BTCUSDT_1h"];
    expect(klines.length).toBeLessThanOrEqual(500);
  });

  it("fetchKlines fetches from Binance API", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => [
        [1000, "95000", "96000", "94000", "95500", "100", 2000, "9500000", 5000, "0", "0", "0"],
        [4600, "95500", "96500", "94500", "96000", "120", 5600, "11520000", 6000, "0", "0", "0"],
      ],
    });

    const { useMarketStore } = await import("@/stores/marketStore");
    await useMarketStore.getState().fetchKlines("BTCUSDT", "1h");

    expect(mockFetch).toHaveBeenCalledWith(expect.stringContaining("klines?symbol=BTCUSDT&interval=1h"));
    const klines = useMarketStore.getState().klines["BTCUSDT_1h"];
    expect(klines).toBeDefined();
    expect(klines.length).toBe(2);
  });

  it("fetchKlines handles API errors gracefully", async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 500 });

    const { useMarketStore } = await import("@/stores/marketStore");
    // Should not throw
    await useMarketStore.getState().fetchKlines("BTCUSDT", "1h");
  });
});
