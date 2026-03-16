import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import { Tooltip } from "@/components/ui/tooltip";

describe("Tooltip", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("does not show tooltip content initially", () => {
    render(<Tooltip content="Help text"><button>Hover me</button></Tooltip>);
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });

  it("shows tooltip on mouse enter after delay", () => {
    render(<Tooltip content="Help text"><button>Hover me</button></Tooltip>);
    fireEvent.mouseEnter(screen.getByText("Hover me").closest("div.relative")!);
    act(() => { vi.advanceTimersByTime(300); });
    expect(screen.getByRole("tooltip")).toHaveTextContent("Help text");
  });

  it("hides tooltip on mouse leave", () => {
    render(<Tooltip content="Help text"><button>Hover me</button></Tooltip>);
    const wrapper = screen.getByText("Hover me").closest("div.relative")!;
    fireEvent.mouseEnter(wrapper);
    act(() => { vi.advanceTimersByTime(300); });
    expect(screen.getByRole("tooltip")).toBeInTheDocument();
    fireEvent.mouseLeave(wrapper);
    act(() => { vi.advanceTimersByTime(200); });
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });

  it("shows tooltip on focus", () => {
    render(<Tooltip content="Focus tip"><button>Focus me</button></Tooltip>);
    fireEvent.focus(screen.getByText("Focus me").closest("div.relative")!);
    act(() => { vi.advanceTimersByTime(300); });
    expect(screen.getByRole("tooltip")).toHaveTextContent("Focus tip");
  });
});
