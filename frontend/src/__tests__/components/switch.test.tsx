import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { Switch } from "@/components/ui/switch";

describe("Switch", () => {
  it("renders as a switch role", () => {
    render(<Switch checked={false} onChange={() => {}} />);
    expect(screen.getByRole("switch")).toBeInTheDocument();
  });

  it("reflects checked state via aria-checked", () => {
    const { rerender } = render(<Switch checked={false} onChange={() => {}} />);
    expect(screen.getByRole("switch")).toHaveAttribute("aria-checked", "false");
    rerender(<Switch checked={true} onChange={() => {}} />);
    expect(screen.getByRole("switch")).toHaveAttribute("aria-checked", "true");
  });

  it("calls onChange with toggled value on click", () => {
    const onChange = vi.fn();
    render(<Switch checked={false} onChange={onChange} />);
    fireEvent.click(screen.getByRole("switch"));
    expect(onChange).toHaveBeenCalledWith(true);
  });

  it("renders label when provided", () => {
    render(<Switch checked={false} onChange={() => {}} label="Dark mode" />);
    expect(screen.getByText("Dark mode")).toBeInTheDocument();
  });

  it("does not call onChange when disabled", () => {
    const onChange = vi.fn();
    render(<Switch checked={false} onChange={onChange} disabled />);
    fireEvent.click(screen.getByRole("switch"));
    expect(onChange).not.toHaveBeenCalled();
  });
});
