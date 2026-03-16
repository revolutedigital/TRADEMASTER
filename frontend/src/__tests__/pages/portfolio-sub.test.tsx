import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";

let mockPathname = "/portfolio/fees";
vi.mock("next/navigation", () => ({
  usePathname: () => mockPathname,
}));

vi.mock("next/link", () => ({
  default: ({ children, href, ...props }: { children: React.ReactNode; href: string; [key: string]: unknown }) => (
    <a href={href} {...props}>{children}</a>
  ),
}));

vi.mock("@/lib/utils", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/utils")>();
  return {
    ...actual,
    apiFetch: vi.fn().mockResolvedValue({
      total_fees: 50,
      fees_by_symbol: { BTCUSDT: 30, ETHUSDT: 20 },
      fee_impact_pct: 0.005,
      monthly_fees: [],
      optimal_weights: { BTCUSDT: 0.6, ETHUSDT: 0.4 },
      expected_return: 0.15,
      expected_risk: 0.1,
      sharpe_ratio: 1.5,
      frontier: [],
    }),
  };
});

import FeesPage from "@/app/portfolio/fees/page";
import OptimizerPage from "@/app/portfolio/optimizer/page";

describe("FeesPage", () => {
  it("renders heading", () => {
    render(<FeesPage />);
    expect(screen.getByText(/Taxas|An.*lise de Taxas/)).toBeInTheDocument();
  });
});

describe("OptimizerPage", () => {
  it("renders heading", () => {
    mockPathname = "/portfolio/optimizer";
    render(<OptimizerPage />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(/Otimizador/)
  });

  it("renders optimize button", () => {
    mockPathname = "/portfolio/optimizer";
    render(<OptimizerPage />);
    expect(screen.getByText(/Otimizar|Executar/)).toBeInTheDocument();
  });
});
