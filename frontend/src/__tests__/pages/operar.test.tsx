import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

const mockApiFetch = vi.hoisted(() => vi.fn());

vi.mock("@/lib/utils", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/utils")>();
  return {
    ...actual,
    apiFetch: mockApiFetch,
  };
});

import OperarPage from "@/app/operar/page";

const approvedStudy = {
  symbol: "BTCUSDT",
  execution_mode: "TESTNET",
  market_study: {
    trend: "UPTREND" as const,
    volatility_pct: 2.4,
    liquidity_quote_volume_24h: 120_000_000,
    candles: 4_000,
  },
  pattern_study: {
    regime: "UPTREND" as const,
    pattern: "TREND_CONTINUATION" as const,
    confidence: 0.74,
    relative_volume: 1.24,
    taker_buy_imbalance: 0.18,
    flow_data_available: true,
    explanation: "Tendência e fluxo confirmam continuação.",
  },
  predictive_model: {
    trained: true,
    validation_accuracy: 0.61,
    samples: 3_900,
    latest_signal: "BUY" as const,
  },
  recommendation: {
    strategy_name: "SMA + RSI",
    backtest_id: 41,
    deployment_id: 12,
    deployment_status: "APPROVED" as const,
    reasons: ["Walk-forward aprovado"],
  },
};

const completedStudyJob = {
  id: 11,
  symbol: "BTCUSDT",
  status: "COMPLETED" as const,
  message: "Estudo concluído. Nenhuma estratégia foi ativada.",
  study: approvedStudy,
  error_message: null,
};

const completedMarketScan = {
  id: 71,
  status: "COMPLETED" as const,
  total_assets: 240,
  screened_assets: 240,
  shortlisted_assets: 6,
  studied_assets: 6,
  failed_assets: 4,
  message: "Varredura concluída: 6 ativos passaram pelo estudo completo. Nenhuma estratégia foi ativada.",
  candidates: [
    {
      rank: 1,
      symbol: "SOLUSDT",
      screening_score: 82.5,
      market_trend: "UPTREND" as const,
      price_change_pct_24h: 2.4,
      quote_volume_24h: 80_000_000,
      status: "APPROVED" as const,
      study: { ...approvedStudy, symbol: "SOLUSDT" },
      error_message: null,
    },
  ],
};

describe("OperarPage", () => {
  beforeEach(() => {
    mockApiFetch.mockReset();
    mockApiFetch.mockImplementation((path: string) => {
      if (path.startsWith("/api/v1/asset-intelligence/universe")) {
        return Promise.resolve({
          assets: [
            { symbol: "BTCUSDT", base_asset: "BTC", quote_asset: "USDT", quote_volume_24h: 120_000_000, price_change_pct_24h: 1.2 },
            { symbol: "SOLUSDT", base_asset: "SOL", quote_asset: "USDT", quote_volume_24h: 80_000_000, price_change_pct_24h: -0.4 },
          ],
        });
      }
      if (path === "/api/v1/trading/live/status") {
        return Promise.resolve({ execution_mode: "TESTNET" });
      }
      if (path === "/api/v1/asset-intelligence/studies") {
        return Promise.resolve(completedStudyJob);
      }
      if (path === "/api/v1/asset-intelligence/opportunity-scans") {
        return Promise.resolve(completedMarketScan);
      }
      return Promise.resolve({});
    });
  });

  it("presents one guided flow with a liquid asset catalog", async () => {
    render(<OperarPage />);

    expect(await screen.findByText("Escolha um ativo. A IA faz o resto.")).toBeInTheDocument();
    expect(screen.getByText("Testnet protegido")).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /BTC\/USDT/ })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /SOL\/USDT/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Estudar BTC/USDT" })).toBeInTheDocument();
  });

  it("studies the selected asset before allowing Testnet activation", async () => {
    render(<OperarPage />);

    const studyButton = await screen.findByRole("button", { name: "Estudar BTC/USDT" });
    fireEvent.click(studyButton);

    expect(await screen.findByText("Modelo preditivo")).toBeInTheDocument();
    expect(screen.getByText("SMA + RSI")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Ativar e acompanhar" })).toBeInTheDocument();
    expect(mockApiFetch).toHaveBeenCalledWith(
      "/api/v1/asset-intelligence/studies",
      expect.objectContaining({ method: "POST", body: JSON.stringify({ symbol: "BTCUSDT" }) }),
    );
  });

  it("activates only the validated recommendation and then starts the engine", async () => {
    render(<OperarPage />);
    fireEvent.click(await screen.findByRole("button", { name: "Estudar BTC/USDT" }));
    fireEvent.click(await screen.findByRole("button", { name: "Ativar e acompanhar" }));

    await waitFor(() => {
      expect(mockApiFetch).toHaveBeenCalledWith(
        "/api/v1/strategy-deployments/12/activate",
        expect.objectContaining({ method: "POST" }),
      );
      expect(mockApiFetch).toHaveBeenCalledWith("/api/v1/trading/engine/start", { method: "POST" });
    });
  });

  it("scans the market and lets the operator open a finalist study without activation", async () => {
    render(<OperarPage />);

    fireEvent.click(await screen.findByRole("button", { name: "Buscar oportunidades no mercado" }));

    expect(await screen.findByText("Oportunidades encontradas")).toBeInTheDocument();
    expect(screen.getByText(/#1 SOL\/USDT/)).toBeInTheDocument();
    expect(screen.getByText("Estratégia validada")).toBeInTheDocument();
    expect(mockApiFetch).toHaveBeenCalledWith(
      "/api/v1/asset-intelligence/opportunity-scans",
      expect.objectContaining({ method: "POST" }),
    );

    fireEvent.click(screen.getByRole("button", { name: "Abrir estudo" }));
    expect(await screen.findByText("SMA + RSI")).toBeInTheDocument();
  });
});
