import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

let mockPathname = "/";
vi.mock("next/navigation", () => ({
  usePathname: () => mockPathname,
}));

vi.mock("next/link", () => ({
  default: ({ children, href, ...props }: { children: React.ReactNode; href: string; [key: string]: unknown }) => (
    <a href={href} {...props}>{children}</a>
  ),
}));

vi.mock("@/stores/marketStore", () => ({
  useMarketStore: () => ({
    prices: {
      BTCUSDT: { price: 42000, change_24h: 0.05 },
      ETHUSDT: { price: 2800, change_24h: -0.02 },
    },
  }),
}));

const mockLogout = vi.fn();
vi.mock("@/stores/authStore", () => ({
  useAuthStore: (selector?: (s: unknown) => unknown) => {
    const state = { logout: mockLogout };
    return selector ? selector(state) : state;
  },
}));

vi.mock("@/stores/themeStore", () => ({
  useThemeStore: () => ({
    theme: "dark",
    toggleTheme: vi.fn(),
  }),
}));

vi.mock("@/components/ui/notification-bell", () => ({
  NotificationBell: () => <div data-testid="notif-bell">bell</div>,
}));

vi.mock("@/components/ui/sidebar", () => ({
  MobileSidebar: () => <div data-testid="mobile-sidebar">menu</div>,
}));

vi.mock("@/components/ui/tooltip", () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

import { Header } from "@/components/ui/header";

describe("Header", () => {
  it("renders BTC and ETH symbols", () => {
    render(<Header />);
    expect(screen.getByText("BTC")).toBeInTheDocument();
    expect(screen.getByText("ETH")).toBeInTheDocument();
  });

  it("renders formatted BTC price", () => {
    render(<Header />);
    expect(screen.getByText("$42,000.00")).toBeInTheDocument();
  });

  it("renders Ao Vivo connection status", () => {
    render(<Header />);
    expect(screen.getByText("Ao Vivo")).toBeInTheDocument();
  });

  it("renders logout button (Sair)", () => {
    render(<Header />);
    expect(screen.getByLabelText("Sair")).toBeInTheDocument();
  });

  it("calls logout on Sair click", () => {
    render(<Header />);
    fireEvent.click(screen.getByLabelText("Sair"));
    expect(mockLogout).toHaveBeenCalled();
  });

  it("renders theme toggle button", () => {
    render(<Header />);
    expect(screen.getByLabelText("Modo claro")).toBeInTheDocument();
  });

  it("renders command palette search button", () => {
    render(<Header />);
    expect(screen.getByLabelText("Open command palette")).toBeInTheDocument();
  });

  it("renders mobile sidebar", () => {
    render(<Header />);
    expect(screen.getByTestId("mobile-sidebar")).toBeInTheDocument();
  });

  it("renders notification bell", () => {
    render(<Header />);
    expect(screen.getByTestId("notif-bell")).toBeInTheDocument();
  });
});
