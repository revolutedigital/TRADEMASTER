import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import Loading from "@/app/loading";
import TradingLoading from "@/app/trading/loading";
import PortfolioLoading from "@/app/portfolio/loading";
import BacktestLoading from "@/app/backtest/loading";
import SignalsLoading from "@/app/signals/loading";
import SettingsLoading from "@/app/settings/loading";

describe("Loading (global)", () => {
  it("renders loading spinner and text", () => {
    render(<Loading />);
    expect(screen.getByText("Loading...")).toBeInTheDocument();
  });
});

describe("TradingLoading", () => {
  it("renders skeleton elements", () => {
    const { container } = render(<TradingLoading />);
    const skeletons = container.querySelectorAll("[aria-hidden='true']");
    expect(skeletons.length).toBeGreaterThan(0);
  });
});

describe("PortfolioLoading", () => {
  it("renders skeleton elements", () => {
    const { container } = render(<PortfolioLoading />);
    const skeletons = container.querySelectorAll("[aria-hidden='true']");
    expect(skeletons.length).toBeGreaterThan(0);
  });
});

describe("BacktestLoading", () => {
  it("renders skeleton elements", () => {
    const { container } = render(<BacktestLoading />);
    const skeletons = container.querySelectorAll("[aria-hidden='true']");
    expect(skeletons.length).toBeGreaterThan(0);
  });
});

describe("SignalsLoading", () => {
  it("renders skeleton elements", () => {
    const { container } = render(<SignalsLoading />);
    const skeletons = container.querySelectorAll("[aria-hidden='true']");
    expect(skeletons.length).toBeGreaterThan(0);
  });
});

describe("SettingsLoading", () => {
  it("renders skeleton elements", () => {
    const { container } = render(<SettingsLoading />);
    const skeletons = container.querySelectorAll("[aria-hidden='true']");
    expect(skeletons.length).toBeGreaterThan(0);
  });
});
