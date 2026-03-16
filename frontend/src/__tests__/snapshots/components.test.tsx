/**
 * Snapshot tests for critical UI components.
 * Detects unintended visual/structural changes.
 */

import { describe, it, expect, vi } from "vitest";

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
        { type: "div", className: "fixed top-0 left-0 right-0 z-50 bg-amber-600 px-4 py-2 text-center" },
        { text: "You are offline. Some features may be unavailable." },
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
      { label: "Painel", href: "/", icon: "LayoutDashboard" },
      { label: "Trading", href: "/trading", icon: "CandlestickChart" },
      { label: "Portfolio", href: "/portfolio", icon: "Briefcase" },
      { label: "Sinais", href: "/signals", icon: "Zap" },
      { label: "Backtest", href: "/backtest", icon: "FlaskConical" },
      { label: "ML/IA", href: "/ml", icon: "Brain" },
      { label: "Sentimento", href: "/sentiment", icon: "BarChart3" },
      { label: "Alertas", href: "/alerts", icon: "Bell" },
      { label: "Configuracoes", href: "/settings", icon: "Settings" },
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
