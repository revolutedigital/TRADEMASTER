import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import TradingError from "@/app/trading/error";
import PortfolioError from "@/app/portfolio/error";
import BacktestError from "@/app/backtest/error";
import SignalsError from "@/app/signals/error";
import SettingsError from "@/app/settings/error";

describe("Route Error Pages", () => {
  it("TradingError renders error and Try Again", () => {
    const reset = vi.fn();
    render(<TradingError error={new Error("Trade error")} reset={reset} />);
    expect(screen.getByText("Trading Error")).toBeInTheDocument();
    expect(screen.getByText("Trade error")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Try Again"));
    expect(reset).toHaveBeenCalled();
  });

  it("PortfolioError renders error and reset", () => {
    const reset = vi.fn();
    render(<PortfolioError error={new Error("Portfolio error")} reset={reset} />);
    expect(screen.getByText("Portfolio error")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Try Again"));
    expect(reset).toHaveBeenCalled();
  });

  it("BacktestError renders error and reset", () => {
    const reset = vi.fn();
    render(<BacktestError error={new Error("Backtest error")} reset={reset} />);
    expect(screen.getByText("Backtest error")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Try Again"));
    expect(reset).toHaveBeenCalled();
  });

  it("SignalsError renders error and reset", () => {
    const reset = vi.fn();
    render(<SignalsError error={new Error("Signals error")} reset={reset} />);
    expect(screen.getByText("Signals error")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Try Again"));
    expect(reset).toHaveBeenCalled();
  });

  it("SettingsError renders error and reset", () => {
    const reset = vi.fn();
    render(<SettingsError error={new Error("Settings error")} reset={reset} />);
    expect(screen.getByText("Settings error")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Try Again"));
    expect(reset).toHaveBeenCalled();
  });
});
