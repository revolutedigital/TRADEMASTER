import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

vi.mock("@/lib/utils", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/utils")>();
  return {
    ...actual,
    apiFetch: vi.fn((url: string) =>
      Promise.resolve(
        url === "/api/v1/trading/live/status"
          ? { execution_mode: "PAPER" }
          : {
              total_trades: 0,
              win_rate: 0,
              total_return_pct: 0,
              sharpe_ratio: 0,
              max_drawdown: 0,
              equity_curve: [],
            },
      ),
    ),
  };
});

import StrategyBuilderPage from "@/app/trading/strategy-builder/page";
import { apiFetch } from "@/lib/utils";

async function renderStrategyBuilder() {
  render(<StrategyBuilderPage />);
  await waitFor(() =>
    expect(apiFetch).toHaveBeenCalledWith("/api/v1/trading/live/status"),
  );
}

describe("StrategyBuilderPage", () => {
  it("renders heading", async () => {
    await renderStrategyBuilder();
    expect(screen.getByText("Criador de Estratégias")).toBeInTheDocument();
  });

  it("renders indicator toggles", async () => {
    await renderStrategyBuilder();
    expect(
      screen.getByText(/SMA \(Simple Moving Average\)/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/RSI \(Relative Strength Index\)/),
    ).toBeInTheDocument();
  });

  it("renders run strategy button", async () => {
    await renderStrategyBuilder();
    expect(screen.getByText(/Rodar Backtest/)).toBeInTheDocument();
  });
});
