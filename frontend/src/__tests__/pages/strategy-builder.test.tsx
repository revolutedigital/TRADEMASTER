import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";

vi.mock("@/lib/utils", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/utils")>();
  return {
    ...actual,
    apiFetch: vi.fn().mockResolvedValue({ total_trades: 0, win_rate: 0, total_return_pct: 0, sharpe_ratio: 0, max_drawdown: 0, equity_curve: [] }),
  };
});

import StrategyBuilderPage from "@/app/trading/strategy-builder/page";

describe("StrategyBuilderPage", () => {
  it("renders heading", () => {
    render(<StrategyBuilderPage />);
    expect(screen.getByText("Criador de Estratégias")).toBeInTheDocument();
  });

  it("renders indicator toggles", () => {
    render(<StrategyBuilderPage />);
    expect(screen.getByText(/SMA \(Simple Moving Average\)/)).toBeInTheDocument();
    expect(screen.getByText(/RSI \(Relative Strength Index\)/)).toBeInTheDocument();
  });

  it("renders run strategy button", () => {
    render(<StrategyBuilderPage />);
    expect(screen.getByText(/Rodar Backtest/)).toBeInTheDocument();
  });
});
