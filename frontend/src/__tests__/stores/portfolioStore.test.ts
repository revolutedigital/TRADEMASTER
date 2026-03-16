import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock apiFetch using vi.hoisted to avoid hoisting issues
const mockApiFetch = vi.hoisted(() => vi.fn());

vi.mock("@/lib/utils", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/utils")>();
  return {
    ...actual,
    apiFetch: mockApiFetch,
  };
});

import { usePortfolioStore } from "@/stores/portfolioStore";

describe("portfolioStore", () => {
  beforeEach(() => {
    mockApiFetch.mockReset();
    usePortfolioStore.setState({
      positions: [],
      orders: [],
      signals: [],
      summary: null,
      riskStatus: null,
    });
  });

  it("initializes with empty state", () => {
    const state = usePortfolioStore.getState();
    expect(state.positions).toEqual([]);
    expect(state.orders).toEqual([]);
    expect(state.signals).toEqual([]);
    expect(state.summary).toBeNull();
    expect(state.riskStatus).toBeNull();
  });

  it("fetchPositions populates positions", async () => {
    const mockPositions = [
      { id: "1", symbol: "BTCUSDT", side: "LONG", entry_price: 95000, quantity: 0.1, current_price: 96000, unrealized_pnl: 100, realized_pnl: 0, stop_loss_price: null, take_profit_price: null, opened_at: "2024-01-01" },
    ];
    mockApiFetch.mockResolvedValueOnce(mockPositions);
    await usePortfolioStore.getState().fetchPositions();
    expect(usePortfolioStore.getState().positions).toEqual(mockPositions);
  });

  it("fetchSummary populates summary", async () => {
    const mockSummary = { total_equity: 10000, available_balance: 5000, total_unrealized_pnl: 500, total_realized_pnl: 200, total_exposure: 5000, exposure_pct: 0.5, open_positions: 2, daily_pnl: 150, daily_pnl_pct: 0.015 };
    mockApiFetch.mockResolvedValueOnce(mockSummary);
    await usePortfolioStore.getState().fetchSummary();
    expect(usePortfolioStore.getState().summary).toEqual(mockSummary);
  });

  it("updatePosition updates an existing position", () => {
    const position = { id: "1", symbol: "BTCUSDT", side: "LONG" as const, entry_price: 95000, quantity: 0.1, current_price: 96000, unrealized_pnl: 100, realized_pnl: 0, stop_loss_price: null, take_profit_price: null, opened_at: "2024-01-01" };
    usePortfolioStore.setState({ positions: [position] });

    const updated = { ...position, current_price: 97000, unrealized_pnl: 200 };
    usePortfolioStore.getState().updatePosition(updated);

    expect(usePortfolioStore.getState().positions[0].current_price).toBe(97000);
    expect(usePortfolioStore.getState().positions[0].unrealized_pnl).toBe(200);
  });

  it("addSignal adds signal and caps at 100", () => {
    const signal = { id: "s1", symbol: "BTCUSDT", action: "BUY" as const, strength: 0.8, confidence: 0.75, model_source: "xgboost", created_at: "2024-01-01" };
    usePortfolioStore.getState().addSignal(signal);
    expect(usePortfolioStore.getState().signals.length).toBe(1);
    expect(usePortfolioStore.getState().signals[0].id).toBe("s1");
  });

  it("updateRiskStatus sets risk status", () => {
    const status = { state: "NORMAL" as const, circuit_breaker_state: "NORMAL" as const, can_trade: true, position_size_multiplier: 1, daily_drawdown: 0, weekly_drawdown: 0, monthly_drawdown: 0, max_drawdown: 0.05, peak_equity: 10000 };
    usePortfolioStore.getState().updateRiskStatus(status);
    expect(usePortfolioStore.getState().riskStatus).toEqual(status);
  });
});
