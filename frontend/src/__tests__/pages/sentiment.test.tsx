import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";

let mockPathname = "/sentiment";
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
      fear_greed_index: 65,
      fear_greed_label: "Greed",
      funding_rates: { BTCUSDT: 0.0001, ETHUSDT: 0.0002 },
      long_short_ratio: { BTCUSDT: 1.5, ETHUSDT: 1.2 },
      open_interest: { BTCUSDT: 5000000, ETHUSDT: 2000000 },
    }),
  };
});

import SentimentPage from "@/app/sentiment/page";

describe("SentimentPage", () => {
  it("renders Sentimento de Mercado heading", () => {
    render(<SentimentPage />);
    expect(screen.getByText("Sentimento de Mercado")).toBeInTheDocument();
  });

  it("renders description", () => {
    render(<SentimentPage />);
    expect(screen.getByText(/Indicadores de sentimento/)).toBeInTheDocument();
  });
});
