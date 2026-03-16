import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

// Need to mock PageHeader since it uses next/navigation (Breadcrumbs)
let mockPathname = "/alerts";
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

import AlertsPage from "@/app/alerts/page";

describe("AlertsPage", () => {
  it("renders Alertas de Preco heading", () => {
    render(<AlertsPage />);
    expect(screen.getByText(/Alertas de Pre/)).toBeInTheDocument();
  });

  it("renders new alert button", () => {
    render(<AlertsPage />);
    expect(screen.getByText(/Novo Alerta/)).toBeInTheDocument();
  });

  it("shows empty state when no alerts", async () => {
    render(<AlertsPage />);
    // Wait for loading to complete - the component fetches alerts on mount
    const emptyText = await screen.findByText(/Nenhum alerta configurado/);
    expect(emptyText).toBeInTheDocument();
  });

  it("shows form when new alert button is clicked", () => {
    render(<AlertsPage />);
    fireEvent.click(screen.getByText(/Novo Alerta/));
    expect(screen.getByText("Criar Alerta")).toBeInTheDocument();
  });
});
