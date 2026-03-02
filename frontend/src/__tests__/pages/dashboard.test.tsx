import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";

// Mock hooks
vi.mock("@/hooks/useMarketData", () => ({
  useMarketData: vi.fn(() => ({
    currentPrice: { symbol: "BTCUSDT", price: 95000, change_24h: 0.02, volume_24h: 50000000, high_24h: 96000, low_24h: 93000 },
    currentKlines: [],
    selectedSymbol: "BTCUSDT",
    selectedInterval: "1h",
    setSelectedSymbol: vi.fn(),
    setSelectedInterval: vi.fn(),
    prices: {},
  })),
}));

vi.mock("@/hooks/usePortfolio", () => ({
  usePortfolio: vi.fn(() => ({
    positions: [],
    signals: [],
    summary: { total_equity: 10000, daily_pnl: 150, daily_pnl_pct: 0.015, open_positions: 0, exposure_pct: 0 },
    riskStatus: { circuit_breaker_state: "NORMAL", can_trade: true, daily_drawdown: 0 },
    fetchPositions: vi.fn(),
    fetchSummary: vi.fn(),
  })),
}));

// Mock apiFetch
vi.mock("@/lib/utils", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/utils")>();
  return {
    ...actual,
    apiFetch: vi.fn().mockResolvedValue({ engine_running: true }),
  };
});

// Mock chart component (canvas-based)
vi.mock("@/components/charts/candlestick-chart", () => ({
  CandlestickChart: () => <div data-testid="candlestick-chart">Chart</div>,
}));

import DashboardPage from "@/app/page";

describe("DashboardPage", () => {
  it("renders Dashboard heading", () => {
    render(<DashboardPage />);
    expect(screen.getByText("Dashboard")).toBeInTheDocument();
  });

  it("renders stat cards with portfolio data", () => {
    render(<DashboardPage />);
    expect(screen.getByText("Total Equity")).toBeInTheDocument();
    expect(screen.getByText("Daily P&L")).toBeInTheDocument();
    expect(screen.getByText("Open Positions")).toBeInTheDocument();
    expect(screen.getByText("Risk Status")).toBeInTheDocument();
  });

  it("renders symbol selector buttons", () => {
    render(<DashboardPage />);
    expect(screen.getByText("BTC/USDT")).toBeInTheDocument();
    expect(screen.getByText("ETH/USDT")).toBeInTheDocument();
  });

  it("renders interval selector buttons", () => {
    render(<DashboardPage />);
    expect(screen.getByText("1m")).toBeInTheDocument();
    expect(screen.getByText("5m")).toBeInTheDocument();
    expect(screen.getByText("15m")).toBeInTheDocument();
    expect(screen.getByText("1h")).toBeInTheDocument();
    expect(screen.getByText("4h")).toBeInTheDocument();
    expect(screen.getByText("1d")).toBeInTheDocument();
  });

  it("renders Paper Trading panel", () => {
    render(<DashboardPage />);
    expect(screen.getByText("Paper Trading")).toBeInTheDocument();
    expect(screen.getByText("Simulated")).toBeInTheDocument();
  });

  it("renders Buy and Sell buttons", () => {
    render(<DashboardPage />);
    expect(screen.getByText(/buy.*long/i)).toBeInTheDocument();
    expect(screen.getByText(/sell.*short/i)).toBeInTheDocument();
  });

  it("renders Open Positions section", () => {
    render(<DashboardPage />);
    expect(screen.getByText("Open Positions")).toBeInTheDocument();
    expect(screen.getByText("No open positions")).toBeInTheDocument();
  });

  it("renders Recent Signals section", () => {
    render(<DashboardPage />);
    expect(screen.getByText("Recent Signals")).toBeInTheDocument();
  });

  it("renders risk info in trading panel", () => {
    render(<DashboardPage />);
    expect(screen.getByText("Stop Loss")).toBeInTheDocument();
    expect(screen.getByText("Take Profit")).toBeInTheDocument();
    expect(screen.getByText("Fee")).toBeInTheDocument();
  });

  it("renders candlestick chart", () => {
    render(<DashboardPage />);
    expect(screen.getByTestId("candlestick-chart")).toBeInTheDocument();
  });

  it("shows NORMAL risk status", () => {
    render(<DashboardPage />);
    expect(screen.getByText("NORMAL")).toBeInTheDocument();
  });
});
