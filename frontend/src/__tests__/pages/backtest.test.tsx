import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";

vi.mock("@/lib/utils", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/utils")>();
  return {
    ...actual,
    apiFetch: vi.fn().mockResolvedValue([]),
  };
});

vi.mock("@/components/charts/equity-chart", () => ({
  EquityChart: () => <div data-testid="equity-chart">Chart</div>,
}));

import BacktestPage from "@/app/backtest/page";

describe("BacktestPage", () => {
  it("renders Backtesting heading", () => {
    render(<BacktestPage />);
    expect(screen.getByText("Backtesting")).toBeInTheDocument();
  });

  it("renders symbol and interval selectors", () => {
    render(<BacktestPage />);
    expect(screen.getByText("BTC/USDT")).toBeInTheDocument();
    expect(screen.getByText("ETH/USDT")).toBeInTheDocument();
  });

  it("renders run backtest button", () => {
    render(<BacktestPage />);
    expect(screen.getByText(/Rodar Backtest/)).toBeInTheDocument();
  });

  it("renders configuration card", () => {
    render(<BacktestPage />);
    expect(screen.getByText(/Configura.*Backtest/)).toBeInTheDocument();
  });
});
