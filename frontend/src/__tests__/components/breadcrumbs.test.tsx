import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";

let mockPathname = "/trading/history";

// Mock next/navigation
vi.mock("next/navigation", () => ({
  usePathname: () => mockPathname,
}));

// Mock next/link
vi.mock("next/link", () => ({
  default: ({ children, href, ...props }: { children: React.ReactNode; href: string; [key: string]: unknown }) => (
    <a href={href} {...props}>{children}</a>
  ),
}));

import { Breadcrumbs } from "@/components/ui/breadcrumbs";

describe("Breadcrumbs", () => {
  beforeEach(() => {
    mockPathname = "/trading/history";
  });

  it("renders breadcrumb trail for nested path", () => {
    render(<Breadcrumbs />);
    expect(screen.getByText("Trading")).toBeInTheDocument();
    expect(screen.getByText("History")).toBeInTheDocument();
  });

  it("renders home link", () => {
    render(<Breadcrumbs />);
    expect(screen.getByLabelText("Home")).toBeInTheDocument();
  });

  it("marks last crumb as current page", () => {
    render(<Breadcrumbs />);
    const lastCrumb = screen.getByText("History");
    expect(lastCrumb).toHaveAttribute("aria-current", "page");
  });

  it("returns null on root path", () => {
    mockPathname = "/";
    const { container } = render(<Breadcrumbs />);
    expect(container.innerHTML).toBe("");
  });

  it("renders breadcrumb navigation with correct aria-label", () => {
    mockPathname = "/portfolio/optimizer";
    render(<Breadcrumbs />);
    expect(screen.getByLabelText("Breadcrumb")).toBeInTheDocument();
  });
});
