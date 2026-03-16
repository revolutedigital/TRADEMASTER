import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";

const mockPush = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: mockPush }),
}));

import { CommandPalette } from "@/components/ui/command-palette";

describe("CommandPalette", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("is hidden by default", () => {
    render(<CommandPalette />);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("opens with Ctrl+K", () => {
    render(<CommandPalette />);
    act(() => {
      document.dispatchEvent(new KeyboardEvent("keydown", { key: "k", ctrlKey: true }));
    });
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/comando/i)).toBeInTheDocument();
  });

  it("opens with Meta+K (Cmd+K)", () => {
    render(<CommandPalette />);
    act(() => {
      document.dispatchEvent(new KeyboardEvent("keydown", { key: "k", metaKey: true }));
    });
    expect(screen.getByRole("dialog")).toBeInTheDocument();
  });

  it("shows navigation commands when open", () => {
    render(<CommandPalette />);
    act(() => {
      document.dispatchEvent(new KeyboardEvent("keydown", { key: "k", ctrlKey: true }));
    });
    expect(screen.getByText("Painel")).toBeInTheDocument();
    expect(screen.getByText("Terminal de Trading")).toBeInTheDocument();
    expect(screen.getByText("Configurações")).toBeInTheDocument();
  });

  it("shows trading commands", () => {
    render(<CommandPalette />);
    act(() => {
      document.dispatchEvent(new KeyboardEvent("keydown", { key: "k", ctrlKey: true }));
    });
    expect(screen.getByText("Comprar BTC")).toBeInTheDocument();
    expect(screen.getByText("Vender BTC")).toBeInTheDocument();
  });

  it("filters commands based on query", () => {
    render(<CommandPalette />);
    act(() => {
      document.dispatchEvent(new KeyboardEvent("keydown", { key: "k", ctrlKey: true }));
    });
    const input = screen.getByPlaceholderText(/comando/i);
    fireEvent.change(input, { target: { value: "backtest" } });
    expect(screen.getByText("Backtesting")).toBeInTheDocument();
    expect(screen.queryByText("Comprar BTC")).not.toBeInTheDocument();
  });

  it("shows no results message for unmatched query", () => {
    render(<CommandPalette />);
    act(() => {
      document.dispatchEvent(new KeyboardEvent("keydown", { key: "k", ctrlKey: true }));
    });
    const input = screen.getByPlaceholderText(/comando/i);
    fireEvent.change(input, { target: { value: "zzzzzzzzz" } });
    expect(screen.getByText("Nenhum resultado encontrado")).toBeInTheDocument();
  });

  it("navigates on command click", () => {
    render(<CommandPalette />);
    act(() => {
      document.dispatchEvent(new KeyboardEvent("keydown", { key: "k", ctrlKey: true }));
    });
    fireEvent.click(screen.getByText("Backtesting"));
    expect(mockPush).toHaveBeenCalledWith("/backtest");
  });
});
