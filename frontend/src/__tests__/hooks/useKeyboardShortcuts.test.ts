import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";

// Mock next/navigation
const mockPush = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: mockPush }),
}));

import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";

describe("useKeyboardShortcuts", () => {
  beforeEach(() => {
    mockPush.mockReset();
  });

  it("returns an array of shortcut configs", () => {
    const { result } = renderHook(() => useKeyboardShortcuts());
    expect(result.current).toBeInstanceOf(Array);
    expect(result.current.length).toBeGreaterThan(0);
  });

  it("includes a shortcut for Dashboard (Ctrl+D)", () => {
    const { result } = renderHook(() => useKeyboardShortcuts());
    const dashShortcut = result.current.find((s) => s.key === "d" && s.ctrl);
    expect(dashShortcut).toBeDefined();
    expect(dashShortcut!.description).toContain("Dashboard");
  });

  it("includes a shortcut for Trading (Ctrl+T)", () => {
    const { result } = renderHook(() => useKeyboardShortcuts());
    const tradeShortcut = result.current.find((s) => s.key === "t" && s.ctrl);
    expect(tradeShortcut).toBeDefined();
    expect(tradeShortcut!.description).toContain("Trading");
  });

  it("navigates to Dashboard on Ctrl+D", () => {
    renderHook(() => useKeyboardShortcuts());

    act(() => {
      window.dispatchEvent(
        new KeyboardEvent("keydown", { key: "d", ctrlKey: true, bubbles: true })
      );
    });

    expect(mockPush).toHaveBeenCalledWith("/");
  });

  it("navigates to Trading on Ctrl+T", () => {
    renderHook(() => useKeyboardShortcuts());

    act(() => {
      window.dispatchEvent(
        new KeyboardEvent("keydown", { key: "t", ctrlKey: true, bubbles: true })
      );
    });

    expect(mockPush).toHaveBeenCalledWith("/trading");
  });

  it("navigates to Portfolio on Ctrl+Shift+P", () => {
    renderHook(() => useKeyboardShortcuts());

    act(() => {
      window.dispatchEvent(
        new KeyboardEvent("keydown", {
          key: "p",
          ctrlKey: true,
          shiftKey: true,
          bubbles: true,
        })
      );
    });

    expect(mockPush).toHaveBeenCalledWith("/portfolio");
  });

  it("does not fire shortcuts when typing in an INPUT", () => {
    renderHook(() => useKeyboardShortcuts());

    const input = document.createElement("input");
    document.body.appendChild(input);
    input.focus();

    const event = new KeyboardEvent("keydown", {
      key: "d",
      ctrlKey: true,
      bubbles: true,
    });
    Object.defineProperty(event, "target", { value: input });
    window.dispatchEvent(event);

    // Should NOT navigate since target is an input
    expect(mockPush).not.toHaveBeenCalled();

    document.body.removeChild(input);
  });
});
