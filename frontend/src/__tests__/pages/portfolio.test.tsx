import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";

vi.mock("@/hooks/usePortfolio", () => ({
  usePortfolio: vi.fn(() => ({
    positions: [],
    summary: {
      total_equity: 10000,
      available_balance: 8000,
      total_unrealized_pnl: 200,
      total_realized_pnl: 500,
      total_exposure: 2000,
      exposure_pct: 0.2,
      open_positions: 0,
      daily_pnl: 100,
      daily_pnl_pct: 0.01,
    },
    riskStatus: { circuit_breaker_state: "NORMAL", can_trade: true, daily_drawdown: 0, weekly_drawdown: 0, position_size_multiplier: 1.0 },
  })),
}));

vi.mock("@/components/ui/export-button", () => ({
  ExportButton: () => <button>Export</button>,
}));

import PortfolioPage from "@/app/portfolio/page";

describe("PortfolioPage", () => {
  it("renders Portfolio heading", () => {
    render(<PortfolioPage />);
    expect(screen.getByText(/Portf/)).toBeInTheDocument();
  });

  it("renders stat cards", () => {
    render(<PortfolioPage />);
    expect(screen.getByText(/Patrim.*Total/)).toBeInTheDocument();
    expect(screen.getByText(/Saldo Dispon/)).toBeInTheDocument();
  });

  it("renders risk management section", () => {
    render(<PortfolioPage />);
    expect(screen.getByText(/Gest.*Risco/)).toBeInTheDocument();
    expect(screen.getByText("NORMAL")).toBeInTheDocument();
  });

  it("renders positions table with empty state", () => {
    render(<PortfolioPage />);
    expect(screen.getByText(/Sem posi.*abertas/)).toBeInTheDocument();
  });

  it("shows can trade status", () => {
    render(<PortfolioPage />);
    expect(screen.getByText("Sim")).toBeInTheDocument();
  });
});
