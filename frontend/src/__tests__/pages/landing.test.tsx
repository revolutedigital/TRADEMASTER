import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import LandingPage from "@/app/(public)/page";

describe("LandingPage", () => {
  it("renders TradeMaster brand name", () => {
    render(<LandingPage />);
    expect(screen.getAllByText("TradeMaster").length).toBeGreaterThan(0);
  });

  it("renders hero heading", () => {
    render(<LandingPage />);
    expect(screen.getByText("Inteligência Artificial")).toBeInTheDocument();
  });

  it("renders Entrar link", () => {
    render(<LandingPage />);
    const links = screen.getAllByText("Entrar");
    expect(links.length).toBeGreaterThan(0);
  });

  it("renders Começar CTA button", () => {
    render(<LandingPage />);
    expect(screen.getByText("Começar")).toBeInTheDocument();
  });

  it("renders Saiba Mais link", () => {
    render(<LandingPage />);
    expect(screen.getByText("Saiba Mais")).toBeInTheDocument();
  });

  it("renders all 4 feature cards", () => {
    render(<LandingPage />);
    expect(screen.getByText("Trading com IA")).toBeInTheDocument();
    expect(screen.getByText("Gestão de Risco")).toBeInTheDocument();
    expect(screen.getByText("Paper Trading")).toBeInTheDocument();
    expect(screen.getByText("Análise em Tempo Real")).toBeInTheDocument();
  });

  it("renders footer text", () => {
    render(<LandingPage />);
    expect(screen.getByText("Apenas Paper Trading")).toBeInTheDocument();
  });
});
