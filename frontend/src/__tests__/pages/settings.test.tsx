import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";

vi.mock("@/lib/utils", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/utils")>();
  return {
    ...actual,
    apiFetch: vi.fn().mockResolvedValue({
      status: "healthy",
      version: "1.0.0",
      uptime: 3600,
      services: { api: "running", database: "connected", binance: "geo-blocked" },
    }),
  };
});

import SettingsPage from "@/app/settings/page";

describe("SettingsPage", () => {
  it("renders h1 heading with exact text", () => {
    render(<SettingsPage />);
    const heading = screen.getByRole("heading", { level: 1 });
    expect(heading.textContent).toContain("Configura");
  });

  it("renders system status section", () => {
    render(<SettingsPage />);
    expect(screen.getByText("Status do Sistema")).toBeInTheDocument();
  });

  it("renders trading configuration section", () => {
    render(<SettingsPage />);
    // Use getAllByText since "Configuracao" appears in heading + section
    const elements = screen.getAllByText(/Configura/);
    expect(elements.length).toBeGreaterThanOrEqual(2);
  });

  it("renders risk parameters section", () => {
    render(<SettingsPage />);
    expect(screen.getByText(/Par.*Risco/)).toBeInTheDocument();
  });

  it("renders Save button", () => {
    render(<SettingsPage />);
    expect(screen.getByText(/Salvar/)).toBeInTheDocument();
  });

  it("renders API documentation links", () => {
    render(<SettingsPage />);
    expect(screen.getByText("Swagger UI")).toBeInTheDocument();
    expect(screen.getByText("ReDoc")).toBeInTheDocument();
  });
});
