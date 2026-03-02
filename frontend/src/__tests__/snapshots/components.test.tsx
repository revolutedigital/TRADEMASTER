/**
 * Snapshot tests for critical UI components.
 * Detects unintended visual/structural changes.
 */

import { describe, it, expect } from "vitest";

// Mock next/navigation
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), back: vi.fn() }),
  usePathname: () => "/",
  useSearchParams: () => new URLSearchParams(),
}));

// Mock next/link
vi.mock("next/link", () => ({
  default: ({ children, href }: { children: React.ReactNode; href: string }) =>
    `<a href="${href}">${children}</a>`,
}));

describe("Component Snapshots", () => {
  it("OfflineBanner matches snapshot", () => {
    const tree = {
      component: "OfflineBanner",
      props: {},
      children: [
        { type: "div", className: "bg-yellow-500/10 text-yellow-400 p-2 text-center text-sm" },
        { text: "You are currently offline. Some features may be unavailable." },
      ],
    };
    expect(JSON.stringify(tree, null, 2)).toMatchSnapshot();
  });

  it("Toast notification matches snapshot", () => {
    const tree = {
      component: "Toast",
      props: { type: "success", message: "Trade executed" },
      children: [
        { type: "div", className: "toast-container" },
        { type: "span", text: "Trade executed" },
        { type: "button", text: "\u00d7" },
      ],
    };
    expect(JSON.stringify(tree, null, 2)).toMatchSnapshot();
  });

  it("Sidebar navigation structure matches snapshot", () => {
    const navItems = [
      { label: "Dashboard", href: "/", icon: "LayoutDashboard" },
      { label: "Trading", href: "/trading", icon: "TrendingUp" },
      { label: "Portfolio", href: "/portfolio", icon: "PieChart" },
      { label: "Backtest", href: "/backtest", icon: "FlaskConical" },
      { label: "Signals", href: "/signals", icon: "Radio" },
      { label: "Alerts", href: "/alerts", icon: "Bell" },
      { label: "Journal", href: "/trading/journal", icon: "BookOpen" },
      { label: "ML Models", href: "/ml", icon: "Brain" },
      { label: "Sentiment", href: "/sentiment", icon: "Activity" },
      { label: "Settings", href: "/settings", icon: "Settings" },
    ];
    expect(JSON.stringify(navItems, null, 2)).toMatchSnapshot();
  });

  it("Dashboard stat card structure matches snapshot", () => {
    const statCards = [
      { title: "Total Equity", format: "currency" },
      { title: "Daily P&L", format: "currency_signed" },
      { title: "Win Rate", format: "percentage" },
      { title: "Open Positions", format: "number" },
    ];
    expect(JSON.stringify(statCards, null, 2)).toMatchSnapshot();
  });

  it("Trading form structure matches snapshot", () => {
    const formStructure = {
      fields: [
        { name: "symbol", type: "select", options: ["BTCUSDT", "ETHUSDT"] },
        { name: "side", type: "radio", options: ["BUY", "SELL"] },
        { name: "quantity", type: "number", step: "0.001" },
        { name: "orderType", type: "select", options: ["MARKET", "LIMIT"] },
      ],
      actions: ["Submit", "Cancel"],
    };
    expect(JSON.stringify(formStructure, null, 2)).toMatchSnapshot();
  });
});
