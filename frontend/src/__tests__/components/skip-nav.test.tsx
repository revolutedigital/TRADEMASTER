import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { SkipNav } from "@/components/ui/skip-nav";

describe("SkipNav", () => {
  it("renders a link to skip to main content", () => {
    render(<SkipNav />);
    const link = screen.getByText("Skip to main content");
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute("href", "#main-content");
  });

  it("is an anchor element", () => {
    render(<SkipNav />);
    expect(screen.getByText("Skip to main content").tagName).toBe("A");
  });
});
