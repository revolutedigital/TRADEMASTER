import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";

let mockPathname = "/backtest/compare";
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
    apiFetch: vi.fn().mockResolvedValue([]),
  };
});

import BacktestComparePage from "@/app/backtest/compare/page";

describe("BacktestComparePage", () => {
  it("renders heading", () => {
    render(<BacktestComparePage />);
    expect(screen.getByText(/Comparar Backtests/)).toBeInTheDocument();
  });
});
