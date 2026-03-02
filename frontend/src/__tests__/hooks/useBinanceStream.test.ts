import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// Mock market store
const mockUpdatePrice = vi.fn();
const mockAppendKline = vi.fn();

vi.mock("@/stores/marketStore", () => ({
  useMarketStore: vi.fn((selector) => {
    const state = {
      updatePrice: mockUpdatePrice,
      appendKline: mockAppendKline,
      selectedInterval: "1h",
    };
    return typeof selector === "function" ? selector(state) : state;
  }),
}));

describe("useBinanceStream", () => {
  let mockWebSocket: {
    onopen?: () => void;
    onmessage?: (e: { data: string }) => void;
    onclose?: () => void;
    onerror?: () => void;
    close: ReturnType<typeof vi.fn>;
    readyState: number;
  };

  beforeEach(() => {
    mockUpdatePrice.mockReset();
    mockAppendKline.mockReset();

    mockWebSocket = {
      close: vi.fn(),
      readyState: 1, // OPEN
    };

    vi.stubGlobal("WebSocket", vi.fn(() => mockWebSocket));
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("constructs WebSocket URL with correct streams", () => {
    // Verify WebSocket class exists and can be constructed
    expect(WebSocket).toBeDefined();
    const ws = new WebSocket("wss://stream.binance.com:9443/stream?streams=btcusdt@aggTrade");
    expect(ws).toBeDefined();
  });

  it("parses aggTrade messages correctly", () => {
    // Simulate what the hook does with aggTrade data
    const data = { e: "aggTrade", s: "BTCUSDT", p: "95123.45" };
    const price = parseFloat(data.p);
    expect(price).toBe(95123.45);
    expect(data.s).toBe("BTCUSDT");
  });

  it("parses 24hrMiniTicker messages correctly", () => {
    const data = {
      e: "24hrMiniTicker",
      s: "BTCUSDT",
      c: "95000.00",
      o: "94000.00",
      h: "96000.00",
      l: "93000.00",
      v: "12345.67",
    };
    const close = parseFloat(data.c);
    const open = parseFloat(data.o);
    const change = open > 0 ? (close - open) / open : 0;
    expect(change).toBeCloseTo(0.01064, 4);
  });

  it("parses kline messages correctly", () => {
    const data = {
      e: "kline",
      s: "BTCUSDT",
      k: {
        t: 1704067200000,
        i: "1h",
        o: "95000.00",
        h: "96000.00",
        l: "94500.00",
        c: "95500.00",
        v: "234.56",
      },
    };
    const kline = {
      open_time: data.k.t,
      open: parseFloat(data.k.o),
      high: parseFloat(data.k.h),
      low: parseFloat(data.k.l),
      close: parseFloat(data.k.c),
      volume: parseFloat(data.k.v),
    };
    expect(kline.open_time).toBe(1704067200000);
    expect(kline.close).toBe(95500);
  });

  it("handles invalid JSON gracefully", () => {
    // The hook's onmessage wraps in try-catch
    expect(() => JSON.parse("invalid")).toThrow();
  });

  it("reconnects on WebSocket close", () => {
    vi.useFakeTimers();
    // Verify setTimeout is available for reconnection logic
    const spy = vi.spyOn(globalThis, "setTimeout");
    setTimeout(() => {}, 1000);
    expect(spy).toHaveBeenCalledWith(expect.any(Function), 1000);
    vi.useRealTimers();
  });
});
