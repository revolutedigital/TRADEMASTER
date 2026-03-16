import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ShortcutHelp } from "@/components/ui/shortcut-help";

const mockShortcuts = [
  { key: "d", ctrl: true, description: "Go to Dashboard" },
  { key: "t", ctrl: true, description: "Go to Trading" },
  { key: "p", ctrl: true, shift: true, description: "Go to Portfolio" },
];

describe("ShortcutHelp", () => {
  it("does not render when isOpen is false", () => {
    const { container } = render(
      <ShortcutHelp isOpen={false} onClose={vi.fn()} shortcuts={mockShortcuts} />
    );
    expect(container.innerHTML).toBe("");
  });

  it("renders when isOpen is true", () => {
    render(
      <ShortcutHelp isOpen={true} onClose={vi.fn()} shortcuts={mockShortcuts} />
    );
    expect(screen.getByText("Keyboard Shortcuts")).toBeInTheDocument();
  });

  it("renders all shortcuts with descriptions", () => {
    render(
      <ShortcutHelp isOpen={true} onClose={vi.fn()} shortcuts={mockShortcuts} />
    );
    expect(screen.getByText("Go to Dashboard")).toBeInTheDocument();
    expect(screen.getByText("Go to Trading")).toBeInTheDocument();
    expect(screen.getByText("Go to Portfolio")).toBeInTheDocument();
  });

  it("formats key combinations correctly", () => {
    render(
      <ShortcutHelp isOpen={true} onClose={vi.fn()} shortcuts={mockShortcuts} />
    );
    // Ctrl + D
    expect(screen.getByText("Ctrl + D")).toBeInTheDocument();
    // Ctrl + T
    expect(screen.getByText("Ctrl + T")).toBeInTheDocument();
    // Ctrl + Shift + P
    expect(screen.getByText("Ctrl + Shift + P")).toBeInTheDocument();
  });

  it("shows the built-in '?' shortcut for help", () => {
    render(
      <ShortcutHelp isOpen={true} onClose={vi.fn()} shortcuts={mockShortcuts} />
    );
    expect(screen.getByText("Show this help")).toBeInTheDocument();
    expect(screen.getByText("?")).toBeInTheDocument();
  });

  it("calls onClose when ESC button is clicked", () => {
    const onClose = vi.fn();
    render(
      <ShortcutHelp isOpen={true} onClose={onClose} shortcuts={mockShortcuts} />
    );
    fireEvent.click(screen.getByText("ESC"));
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("calls onClose when backdrop is clicked", () => {
    const onClose = vi.fn();
    render(
      <ShortcutHelp isOpen={true} onClose={onClose} shortcuts={mockShortcuts} />
    );
    // Click on the backdrop (outermost div)
    const backdrop = screen.getByText("Keyboard Shortcuts").closest(".fixed");
    fireEvent.click(backdrop!);
    expect(onClose).toHaveBeenCalledOnce();
  });
});
