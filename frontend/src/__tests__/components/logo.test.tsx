import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { Logo } from "@/components/ui/logo";

describe("Logo", () => {
  it("renders full variant by default", () => {
    render(<Logo />);
    expect(screen.getByLabelText("TradeMaster")).toBeInTheDocument();
    expect(screen.getByText("Trade")).toBeInTheDocument();
    expect(screen.getByText("Master")).toBeInTheDocument();
  });

  it("renders icon-only variant", () => {
    const { container } = render(<Logo variant="icon" />);
    const svg = container.querySelector("svg");
    expect(svg).toBeTruthy();
    // Should not have text
    expect(screen.queryByText("Trade")).toBeNull();
  });

  it("renders with sm size class on container", () => {
    render(<Logo size="sm" />);
    const container = screen.getByLabelText("TradeMaster");
    expect(container.className).toContain("gap-1.5");
  });

  it("renders with lg size class on container", () => {
    render(<Logo size="lg" />);
    const container = screen.getByLabelText("TradeMaster");
    expect(container.className).toContain("gap-2.5");
  });

  it("svg is aria-hidden", () => {
    const { container } = render(<Logo />);
    const svg = container.querySelector("svg");
    expect(svg?.getAttribute("aria-hidden")).toBe("true");
  });
});
