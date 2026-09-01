import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";

const mockApiFetch = vi.hoisted(() => vi.fn());

vi.mock("@/lib/utils", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/utils")>();
  return { ...actual, apiFetch: mockApiFetch };
});

import HomePage from "@/app/page";

describe("HomePage", () => {
  beforeEach(() => {
    mockApiFetch.mockReset();
    mockApiFetch.mockImplementation((path: string) => {
      if (path.startsWith("/api/v1/asset-intelligence/universe")) {
        return Promise.resolve({
          assets: [
            { symbol: "BTCUSDT", base_asset: "BTC", quote_asset: "USDT", quote_volume_24h: 100_000_000, price_change_pct_24h: 1.2 },
          ],
        });
      }
      return Promise.resolve({ execution_mode: "TESTNET" });
    });
  });

  it("uses the guided operation workflow as the default page", async () => {
    render(<HomePage />);

    expect(await screen.findByText("Escolha um ativo. A IA faz o resto.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Estudar BTC/USDT" })).toBeInTheDocument();
    expect(screen.queryByText("Paper Trading")).not.toBeInTheDocument();
    expect(screen.queryByText("Comprar")).not.toBeInTheDocument();
  });
});
