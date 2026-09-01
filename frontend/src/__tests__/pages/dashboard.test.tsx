import { beforeEach, describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

const runtimeMode = vi.hoisted(() => ({ value: "PAPER" }));

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

vi.mock("@/stores/onboardingStore", () => ({
  useOnboardingStore: vi.fn((selector: (s: { completed: boolean }) => boolean) =>
    selector({ completed: true })
  ),
}));

// Mock apiFetch
vi.mock("@/lib/utils", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/utils")>();
  return {
    ...actual,
    apiFetch: vi.fn((path: string) =>
      Promise.resolve(
        path === "/api/v1/trading/live/status"
          ? { execution_mode: runtimeMode.value }
          : { engine_running: true },
      ),
    ),
  };
});

// Mock chart component (canvas-based)
vi.mock("@/components/charts/candlestick-chart", () => ({
  CandlestickChart: () => <div data-testid="candlestick-chart">Chart</div>,
}));

// Mock LivePrice
vi.mock("@/components/ui/live-price", () => ({
  LivePrice: ({ price }: { price: number }) => <span>{price}</span>,
}));

// Mock OnboardingWizard
vi.mock("@/components/onboarding/wizard", () => ({
  OnboardingWizard: () => null,
}));

import DashboardPage from "@/app/page";

describe("DashboardPage", () => {
  beforeEach(() => {
    runtimeMode.value = "PAPER";
  });

  it("renders Painel heading", () => {
    render(<DashboardPage />);
    expect(screen.getByText("Painel")).toBeInTheDocument();
  });

  it("renders stat cards with portfolio data (PT-BR labels)", () => {
    render(<DashboardPage />);
    expect(screen.getByText(/Patrim.*Total/)).toBeInTheDocument();
    expect(screen.getByText(/P&L Di.*rio/)).toBeInTheDocument();
    expect(screen.getByText("Status de Risco")).toBeInTheDocument();
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

  it("renders Paper Trading panel", async () => {
    render(<DashboardPage />);
    await waitFor(() => expect(screen.getByText("Paper Trading")).toBeInTheDocument());
    expect(screen.getByText("PAPER")).toBeInTheDocument();
  });

  it("renders Comprar and Vender buttons", async () => {
    render(<DashboardPage />);
    await waitFor(() => expect(screen.getByText("Comprar")).toBeInTheDocument());
    expect(screen.getByText("Vender")).toBeInTheDocument();
  });

  it("replaces paper-order controls with the Testnet strategy flow", async () => {
    runtimeMode.value = "TESTNET";
    render(<DashboardPage />);

    await waitFor(() => expect(screen.getByText("Execução Testnet")).toBeInTheDocument());
    expect(screen.getByText("Configurar estratégia para BTC/USDT")).toBeInTheDocument();
    expect(screen.queryByText("Paper Trading")).not.toBeInTheDocument();
    expect(screen.queryByText("Comprar")).not.toBeInTheDocument();
  });

  it("renders Posicoes Abertas section", () => {
    render(<DashboardPage />);
    // Use getAllByText since "Posicoes Abertas" may appear as stat card label + section header
    const elements = screen.getAllByText(/Posi.*Abertas/);
    expect(elements.length).toBeGreaterThan(0);
    expect(screen.getByText(/Sem posi.*abertas/)).toBeInTheDocument();
  });

  it("renders Sinais Recentes section", () => {
    render(<DashboardPage />);
    expect(screen.getByText(/Sinais Recentes/)).toBeInTheDocument();
  });

  it("renders risk info in trading panel", async () => {
    render(<DashboardPage />);
    await waitFor(() => expect(screen.getByText("Stop Loss")).toBeInTheDocument());
    expect(screen.getByText("Take Profit")).toBeInTheDocument();
    expect(screen.getByText("Taxa")).toBeInTheDocument();
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
