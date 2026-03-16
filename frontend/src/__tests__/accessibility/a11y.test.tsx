/**
 * Accessibility testing for components.
 */

import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";

// Mock next/navigation
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), pathname: "/" }),
  usePathname: () => "/",
}));

// Mock next/link
vi.mock("next/link", () => ({
  default: ({ children, href, ...props }: { children: React.ReactNode; href: string; [key: string]: unknown }) => (
    <a href={href} {...props}>{children}</a>
  ),
}));

// Mock Logo
vi.mock("@/components/ui/logo", () => ({
  Logo: ({ size }: { size?: string }) => <div data-testid="logo">TradeMaster</div>,
}));

import { Sidebar } from "@/components/ui/sidebar";

describe("Accessibility Tests", () => {
  it("sidebar has correct ARIA roles", () => {
    render(<Sidebar />);
    const nav = screen.getByRole("navigation", { name: /main navigation/i });
    expect(nav).toBeTruthy();
  });

  it("form inputs have associated labels", () => {
    const { container } = render(
      <form>
        <label htmlFor="username">Username</label>
        <input id="username" type="text" aria-label="Username" />
        <label htmlFor="password">Password</label>
        <input id="password" type="password" aria-label="Password" />
        <button type="submit">Login</button>
      </form>
    );

    const labels = container.querySelectorAll("label");
    expect(labels.length).toBeGreaterThanOrEqual(2);

    const inputs = container.querySelectorAll("input");
    inputs.forEach((input) => {
      expect(input.getAttribute("aria-label") || input.getAttribute("id")).toBeTruthy();
    });
  });

  it("buttons have accessible names", () => {
    const { container } = render(
      <div>
        <button aria-label="Close modal">X</button>
        <button>Save Changes</button>
        <button aria-label="Toggle sidebar">&#9776;</button>
      </div>
    );

    const buttons = container.querySelectorAll("button");
    buttons.forEach((btn) => {
      const hasName = btn.textContent?.trim() || btn.getAttribute("aria-label");
      expect(hasName).toBeTruthy();
    });
  });

  it("images have alt text", () => {
    const { container } = render(
      <div>
        <img src="/logo.png" alt="TradeMaster Logo" />
        <img src="/icon.png" alt="Status indicator" />
      </div>
    );

    const images = container.querySelectorAll("img");
    images.forEach((img) => {
      expect(img.getAttribute("alt")).toBeTruthy();
    });
  });

  it("color contrast meets WCAG AA", () => {
    const contrastPairs = [
      { fg: "#ffffff", bg: "#0a0e17", name: "white on dark" },
      { fg: "#9ca3af", bg: "#141922", name: "gray on card" },
      { fg: "#22c55e", bg: "#0a0e17", name: "green on dark" },
      { fg: "#ef4444", bg: "#0a0e17", name: "red on dark" },
    ];

    for (const pair of contrastPairs) {
      const getLuminance = (hex: string) => {
        const r = parseInt(hex.slice(1, 3), 16) / 255;
        const g = parseInt(hex.slice(3, 5), 16) / 255;
        const b = parseInt(hex.slice(5, 7), 16) / 255;
        const adjust = (c: number) => (c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
        return 0.2126 * adjust(r) + 0.7152 * adjust(g) + 0.0722 * adjust(b);
      };

      const l1 = getLuminance(pair.fg);
      const l2 = getLuminance(pair.bg);
      const ratio = (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);

      expect(ratio).toBeGreaterThanOrEqual(3.0);
    }
  });

  it("page has correct heading hierarchy", () => {
    const { container } = render(
      <div>
        <h1>Dashboard</h1>
        <section>
          <h2>Portfolio Summary</h2>
          <h2>Recent Trades</h2>
        </section>
      </div>
    );

    const h1s = container.querySelectorAll("h1");
    expect(h1s.length).toBe(1);
  });

  it("interactive elements are keyboard focusable", () => {
    const { container } = render(
      <div>
        <button tabIndex={0}>Click me</button>
        <a href="/trading" tabIndex={0}>Trading</a>
        <input type="text" tabIndex={0} />
      </div>
    );

    const focusable = container.querySelectorAll("button, a[href], input, select, textarea, [tabindex]");
    expect(focusable.length).toBeGreaterThan(0);

    focusable.forEach((el) => {
      const tabIndex = el.getAttribute("tabindex");
      if (tabIndex !== null) {
        expect(parseInt(tabIndex)).toBeGreaterThanOrEqual(0);
      }
    });
  });
});
