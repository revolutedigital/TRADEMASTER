import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";

vi.mock("@/hooks/usePortfolio", () => ({
  usePortfolio: vi.fn(() => ({
    signals: [],
  })),
}));

import SignalsPage from "@/app/signals/page";

describe("SignalsPage", () => {
  it("renders Sinais de IA heading", () => {
    render(<SignalsPage />);
    expect(screen.getByText(/Sinais de IA/)).toBeInTheDocument();
  });

  it("renders signal legend", () => {
    render(<SignalsPage />);
    expect(screen.getByText(/For.*Sinal/)).toBeInTheDocument();
  });

  it("renders empty state when no signals", () => {
    render(<SignalsPage />);
    expect(screen.getByText(/Nenhum sinal gerado/)).toBeInTheDocument();
  });

  it("renders real-time indicator", () => {
    render(<SignalsPage />);
    expect(screen.getByText("Tempo real")).toBeInTheDocument();
  });

  it("renders signal history table header", () => {
    render(<SignalsPage />);
    expect(screen.getByText(/Hist.*Sinais/)).toBeInTheDocument();
  });
});
