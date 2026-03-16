import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";

let mockPathname = "/ml";
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
    apiFetch: vi.fn().mockResolvedValue({ models: [], features: [] }),
  };
});

import MLDashboardPage from "@/app/ml/page";

describe("MLDashboardPage", () => {
  it("renders ML/IA Dashboard heading", () => {
    render(<MLDashboardPage />);
    expect(screen.getByText(/ML.*IA|Painel ML/)).toBeInTheDocument();
  });

  it("renders model symbol selectors", () => {
    render(<MLDashboardPage />);
    expect(screen.getByText("BTC/USDT")).toBeInTheDocument();
    expect(screen.getByText("ETH/USDT")).toBeInTheDocument();
  });
});
