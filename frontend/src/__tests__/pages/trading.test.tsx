import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";

vi.mock("@/hooks/useMarketData", () => ({
  useMarketData: vi.fn(() => ({
    currentPrice: { symbol: "BTCUSDT", price: 95000, change_24h: 0.02, volume_24h: 50000000, high_24h: 96000, low_24h: 93000 },
    currentKlines: [],
    selectedSymbol: "BTCUSDT",
    selectedInterval: "1h",
    setSelectedSymbol: vi.fn(),
    setSelectedInterval: vi.fn(),
  })),
}));

vi.mock("@/hooks/usePortfolio", () => ({
  usePortfolio: vi.fn(() => ({
    positions: [],
    orders: [],
    riskStatus: { circuit_breaker_state: "NORMAL", can_trade: true },
  })),
}));

vi.mock("@/components/charts/candlestick-chart", () => ({
  CandlestickChart: () => <div data-testid="chart">Chart</div>,
}));

vi.mock("@/components/ui/export-button", () => ({
  ExportButton: () => <button>Export</button>,
}));

import TradingPage from "@/app/trading/page";

describe("TradingPage", () => {
  it("renders Terminal de Trading heading", () => {
    render(<TradingPage />);
    expect(screen.getByText("Terminal de Trading")).toBeInTheDocument();
  });

  it("renders symbol selector buttons", () => {
    render(<TradingPage />);
    expect(screen.getByText("BTC/USDT")).toBeInTheDocument();
    expect(screen.getByText("ETH/USDT")).toBeInTheDocument();
  });

  it("renders interval selectors", () => {
    render(<TradingPage />);
    expect(screen.getByText("1h")).toBeInTheDocument();
    expect(screen.getByText("4h")).toBeInTheDocument();
    expect(screen.getByText("1d")).toBeInTheDocument();
  });

  it("renders positions table with empty state", () => {
    render(<TradingPage />);
    expect(screen.getByText(/Posi.*Ativas/)).toBeInTheDocument();
    expect(screen.getByText(/Sem posi.*ativas/)).toBeInTheDocument();
  });

  it("renders orders section", () => {
    render(<TradingPage />);
    expect(screen.getByText(/Ordens Recentes/)).toBeInTheDocument();
  });

  it("shows NORMAL risk status badge", () => {
    render(<TradingPage />);
    expect(screen.getByText("NORMAL")).toBeInTheDocument();
  });
});
