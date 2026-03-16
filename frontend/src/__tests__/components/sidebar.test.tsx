import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

// Track the mock function so we can change it per test
let mockPathname = "/";

// Mock next/navigation
vi.mock("next/navigation", () => ({
  usePathname: () => mockPathname,
}));

// Mock next/link
vi.mock("next/link", () => ({
  default: ({ children, href, ...props }: { children: React.ReactNode; href: string; [key: string]: unknown }) => (
    <a href={href} {...props}>{children}</a>
  ),
}));

// Mock the Logo component
vi.mock("@/components/ui/logo", () => ({
  Logo: ({ size }: { size?: string }) => <div data-testid="logo" data-size={size}>TradeMaster Logo</div>,
}));

import { Sidebar, MobileSidebar } from "@/components/ui/sidebar";

describe("Sidebar", () => {
  beforeEach(() => {
    mockPathname = "/";
  });

  it("renders the Logo component", () => {
    render(<Sidebar />);
    expect(screen.getByTestId("logo")).toBeInTheDocument();
  });

  it("renders all navigation links (PT-BR labels)", () => {
    render(<Sidebar />);
    expect(screen.getByText("Painel")).toBeInTheDocument();
    expect(screen.getByText("Trading")).toBeInTheDocument();
    expect(screen.getByText("Sinais")).toBeInTheDocument();
    expect(screen.getByText("Backtest")).toBeInTheDocument();
    expect(screen.getByText("ML/IA")).toBeInTheDocument();
    expect(screen.getByText("Alertas")).toBeInTheDocument();
  });

  it("highlights Painel as active on root path", () => {
    render(<Sidebar />);
    const painelLink = screen.getByText("Painel").closest("a");
    expect(painelLink).toHaveAttribute("aria-current", "page");
  });

  it("highlights correct link based on pathname", () => {
    mockPathname = "/signals";
    render(<Sidebar />);
    const sinaisLink = screen.getByText("Sinais").closest("a");
    expect(sinaisLink).toHaveAttribute("aria-current", "page");
  });

  it("renders Modo Testnet status", () => {
    render(<Sidebar />);
    expect(screen.getByText("Modo Testnet")).toBeInTheDocument();
  });

  it("has correct navigation role and label", () => {
    render(<Sidebar />);
    const nav = screen.getByRole("navigation", { name: /main navigation/i });
    expect(nav).toBeInTheDocument();
  });

  it("renders Configuracoes link", () => {
    render(<Sidebar />);
    expect(screen.getByText(/Configura/)).toBeInTheDocument();
  });
});

describe("MobileSidebar", () => {
  beforeEach(() => {
    mockPathname = "/";
  });

  it("renders hamburger button with accessible label", () => {
    render(<MobileSidebar />);
    expect(screen.getByLabelText(/abrir menu/i)).toBeInTheDocument();
  });

  it("opens mobile sidebar on button click", () => {
    render(<MobileSidebar />);
    const btn = screen.getByLabelText(/abrir menu/i);
    fireEvent.click(btn);
    expect(screen.getByRole("dialog")).toBeInTheDocument();
  });

  it("closes mobile sidebar with close button", () => {
    render(<MobileSidebar />);
    fireEvent.click(screen.getByLabelText(/abrir menu/i));
    const closeBtn = screen.getByLabelText(/fechar menu/i);
    fireEvent.click(closeBtn);
    const dialog = screen.getByRole("dialog");
    expect(dialog.className).toContain("-translate-x-full");
  });
});
