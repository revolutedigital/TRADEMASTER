import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";

// Mock portfolioStore
const mockFetchSummary = vi.fn();
const mockFetchPositions = vi.fn();
const mockFetchRiskStatus = vi.fn();
const mockFetchSignals = vi.fn();

vi.mock("@/stores/portfolioStore", () => ({
  usePortfolioStore: vi.fn((selector: (s: Record<string, unknown>) => unknown) => {
    const state = {
      fetchSummary: mockFetchSummary,
      fetchPositions: mockFetchPositions,
      fetchRiskStatus: mockFetchRiskStatus,
      fetchSignals: mockFetchSignals,
    };
    return selector(state);
  }),
}));

// Mock useBinanceStream
vi.mock("@/hooks/useBinanceStream", () => ({
  useBinanceStream: vi.fn(),
}));

import { Providers } from "@/components/providers";

describe("Providers", () => {
  it("renders children", () => {
    render(<Providers><div>App Content</div></Providers>);
    expect(screen.getByText("App Content")).toBeInTheDocument();
  });

  it("calls portfolio fetch functions on mount", () => {
    render(<Providers><div>Content</div></Providers>);
    expect(mockFetchSummary).toHaveBeenCalled();
    expect(mockFetchPositions).toHaveBeenCalled();
    expect(mockFetchRiskStatus).toHaveBeenCalled();
    expect(mockFetchSignals).toHaveBeenCalled();
  });
});
