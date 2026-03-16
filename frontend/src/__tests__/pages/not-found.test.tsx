import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";

// Mock next/link
vi.mock("next/link", () => ({
  default: ({ children, href, ...props }: { children: React.ReactNode; href: string; [key: string]: unknown }) => (
    <a href={href} {...props}>{children}</a>
  ),
}));

import NotFound from "@/app/not-found";

describe("NotFoundPage", () => {
  it("renders 404 heading", () => {
    render(<NotFound />);
    expect(screen.getByText("404")).toBeInTheDocument();
  });

  it("renders page not found message", () => {
    render(<NotFound />);
    expect(screen.getByText("Page not found")).toBeInTheDocument();
  });

  it("renders link back to dashboard", () => {
    render(<NotFound />);
    const link = screen.getByText("Back to Dashboard");
    expect(link.closest("a")).toHaveAttribute("href", "/");
  });
});
