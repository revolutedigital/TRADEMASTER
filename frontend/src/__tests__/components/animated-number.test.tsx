import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { AnimatedNumber } from "@/components/ui/animated-number";

describe("AnimatedNumber", () => {
  it("renders the initial value formatted to 2 decimals", () => {
    render(<AnimatedNumber value={42} />);
    expect(screen.getByText("42.00")).toBeInTheDocument();
  });

  it("uses custom format function", () => {
    render(<AnimatedNumber value={1234.5} format={(n) => `$${n.toFixed(0)}`} />);
    expect(screen.getByText("$1235")).toBeInTheDocument();
  });

  it("applies className to span", () => {
    const { container } = render(<AnimatedNumber value={10} className="custom" />);
    expect(container.querySelector("span.custom")).toBeInTheDocument();
  });

  it("has tabular-nums class for monospace digits", () => {
    const { container } = render(<AnimatedNumber value={10} />);
    expect(container.querySelector("span")?.className).toContain("tabular-nums");
  });
});
