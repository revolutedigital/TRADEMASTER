import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { LivePrice } from "@/components/ui/live-price";

describe("LivePrice", () => {
  it("renders formatted price as currency", () => {
    render(<LivePrice price={42000.5} />);
    expect(screen.getByText("$42,000.50")).toBeInTheDocument();
  });

  it("uses custom decimal places", () => {
    render(<LivePrice price={100} decimals={0} />);
    expect(screen.getByText("$100")).toBeInTheDocument();
  });

  it("has font-mono class", () => {
    const { container } = render(<LivePrice price={100} />);
    expect(container.querySelector("span")?.className).toContain("font-mono");
  });

  it("applies custom className", () => {
    const { container } = render(<LivePrice price={100} className="extra" />);
    expect(container.querySelector("span.extra")).toBeInTheDocument();
  });
});
