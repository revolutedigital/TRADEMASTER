import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";

// Mock next/navigation
vi.mock("next/navigation", () => ({
  usePathname: vi.fn(() => "/"),
}));

// Mock next/link
vi.mock("next/link", () => ({
  default: ({ children, href, ...props }: { children: React.ReactNode; href: string; [key: string]: unknown }) => (
    <a href={href} {...props}>{children}</a>
  ),
}));

import { Sidebar, MobileSidebar } from "@/components/ui/sidebar";

describe("Sidebar", () => {
  it("renders the TradeMaster logo text", () => {
    render(<Sidebar />);
    expect(screen.getByText("TradeMaster")).toBeInTheDocument();
  });

  it("renders all navigation links", () => {
    render(<Sidebar />);
    expect(screen.getByText("Dashboard")).toBeInTheDocument();
    expect(screen.getByText("Trading")).toBeInTheDocument();
    expect(screen.getByText("Portfolio")).toBeInTheDocument();
    expect(screen.getByText("Signals")).toBeInTheDocument();
    expect(screen.getByText("Backtest")).toBeInTheDocument();
    expect(screen.getByText("Settings")).toBeInTheDocument();
  });

  it("highlights Dashboard as active on root path", () => {
    render(<Sidebar />);
    const dashboardLink = screen.getByText("Dashboard").closest("a");
    expect(dashboardLink).toHaveAttribute("aria-current", "page");
  });

  it("highlights correct link based on pathname", () => {
    const { usePathname } = require("next/navigation");
    (usePathname as ReturnType<typeof vi.fn>).mockReturnValue("/trading");

    render(<Sidebar />);
    const tradingLink = screen.getByText("Trading").closest("a");
    expect(tradingLink).toHaveAttribute("aria-current", "page");
  });

  it("renders Testnet Mode status", () => {
    render(<Sidebar />);
    expect(screen.getByText("Testnet Mode")).toBeInTheDocument();
  });

  it("renders API Docs external link", () => {
    render(<Sidebar />);
    expect(screen.getByText("API Docs")).toBeInTheDocument();
  });

  it("has correct navigation role and label", () => {
    render(<Sidebar />);
    const nav = screen.getByRole("navigation", { name: /main navigation/i });
    expect(nav).toBeInTheDocument();
  });
});

describe("MobileSidebar", () => {
  it("renders hamburger button", () => {
    render(<MobileSidebar />);
    expect(screen.getByRole("button", { name: /open navigation/i })).toBeInTheDocument();
  });
});
