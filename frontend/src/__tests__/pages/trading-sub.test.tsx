import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";

let mockPathname = "/trading/history";
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

import TradeHistoryPage from "@/app/trading/history/page";
import JournalPage from "@/app/trading/journal/page";

describe("TradeHistoryPage", () => {
  it("renders heading", () => {
    render(<TradeHistoryPage />);
    expect(screen.getByText(/Hist.*de Opera/)).toBeInTheDocument();
  });

  it("renders filter controls", () => {
    render(<TradeHistoryPage />);
    // Should have filter labels
    expect(screen.getByText("Filtros")).toBeInTheDocument();
    expect(screen.getByText("Aplicar Filtros")).toBeInTheDocument();
  });
});

describe("JournalPage", () => {
  it("renders heading", () => {
    mockPathname = "/trading/journal";
    render(<JournalPage />);
    expect(screen.getByText(/Di.*Trading/)).toBeInTheDocument();
  });

  it("renders new entry button", () => {
    mockPathname = "/trading/journal";
    render(<JournalPage />);
    expect(screen.getByText(/Nova Entrada|Novo/)).toBeInTheDocument();
  });
});
