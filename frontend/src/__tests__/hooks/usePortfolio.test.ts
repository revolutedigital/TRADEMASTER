import { describe, it, expect, vi } from "vitest";
import { renderHook } from "@testing-library/react";

const mockState = {
  positions: [],
  summary: null,
  riskStatus: "NORMAL",
  signals: [],
  fetchPositions: vi.fn(),
  fetchSummary: vi.fn(),
};

vi.mock("@/stores/portfolioStore", () => ({
  usePortfolioStore: () => mockState,
}));

import { usePortfolio } from "@/hooks/usePortfolio";

describe("usePortfolio", () => {
  it("returns portfolio store state", () => {
    const { result } = renderHook(() => usePortfolio());
    expect(result.current.positions).toEqual([]);
    expect(result.current.riskStatus).toBe("NORMAL");
  });

  it("returns store functions", () => {
    const { result } = renderHook(() => usePortfolio());
    expect(typeof result.current.fetchPositions).toBe("function");
    expect(typeof result.current.fetchSummary).toBe("function");
  });
});
