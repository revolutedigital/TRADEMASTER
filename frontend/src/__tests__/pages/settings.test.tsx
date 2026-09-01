import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";

vi.mock("@/lib/utils", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/utils")>();
  return {
    ...actual,
    apiFetch: vi.fn((path: string) => {
      if (path === "/api/v1/system/health") {
        return Promise.resolve({
          status: "healthy",
          version: "1.0.0",
          uptime: 3600,
          services: { api: "running", database: "connected", binance: "geo-blocked" },
        });
      }
      if (path === "/api/v1/settings/") {
        return Promise.resolve({
          trading: {
            trading_mode: "paper",
            symbols: ["BTCUSDT", "ETHUSDT"],
            max_risk_per_trade: 0.02,
            max_total_exposure: 0.6,
          },
          risk: {
            max_risk_per_trade: 0.02,
            max_total_exposure: 0.6,
            max_single_asset: 0.3,
            max_daily_drawdown: 0.03,
            max_weekly_drawdown: 0.07,
            max_monthly_drawdown: 0.1,
            max_total_drawdown: 0.15,
            kelly_fraction: 0.15,
          },
          api_docs_url: "/docs",
        });
      }
      return Promise.resolve({
        execution_mode: "PAPER",
        live_enabled: false,
        armed: false,
        armed_until: null,
        armable: false,
        blockers: ["Paper mode enabled"],
        max_notional_per_order: 100,
        max_daily_notional: 300,
        reconciliation: {
          ready: false,
          state: "UNVERIFIED",
          checked_at: null,
          max_age_seconds: 45,
          issues: [],
        },
        testnet_verification: {
          ready: false,
          state: "UNVERIFIED",
          checked_at: null,
          max_age_seconds: 2_592_000,
          issues: [],
        },
      });
    }),
  };
});

import SettingsPage from "@/app/settings/page";

async function renderSettingsPage() {
  render(<SettingsPage />);
  await screen.findByText("PAPER");
}

describe("SettingsPage", () => {
  it("renders h1 heading with exact text", async () => {
    await renderSettingsPage();
    const heading = screen.getByRole("heading", { level: 1 });
    expect(heading.textContent).toContain("Configura");
  });

  it("renders system status section", async () => {
    await renderSettingsPage();
    expect(screen.getByText("Status do Sistema")).toBeInTheDocument();
  });

  it("renders trading configuration section", async () => {
    await renderSettingsPage();
    // Use getAllByText since "Configuracao" appears in heading + section
    const elements = screen.getAllByText(/Configura/);
    expect(elements.length).toBeGreaterThanOrEqual(2);
  });

  it("renders the effective risk limits section", async () => {
    await renderSettingsPage();
    expect(screen.getByText("Limites de Risco Efetivos")).toBeInTheDocument();
  });

  it("renders the effective risk policy without a misleading save control", async () => {
    await renderSettingsPage();
    expect(screen.getByText("Limites de Risco Efetivos")).toBeInTheDocument();
    expect(screen.queryByText(/Salvar/)).not.toBeInTheDocument();
  });

  it("renders API documentation links", async () => {
    await renderSettingsPage();
    expect(screen.getByText("Swagger UI")).toBeInTheDocument();
    expect(screen.getByText("ReDoc")).toBeInTheDocument();
  });
});
