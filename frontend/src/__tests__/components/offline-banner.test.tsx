import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, act } from "@testing-library/react";
import { OfflineBanner } from "@/components/ui/offline-banner";

describe("OfflineBanner", () => {
  let originalOnLine: boolean;

  beforeEach(() => {
    originalOnLine = navigator.onLine;
  });

  afterEach(() => {
    // Restore original onLine value
    Object.defineProperty(navigator, "onLine", {
      value: originalOnLine,
      writable: true,
      configurable: true,
    });
  });

  it("does not render when online", () => {
    Object.defineProperty(navigator, "onLine", { value: true, writable: true, configurable: true });
    const { container } = render(<OfflineBanner />);
    expect(container.innerHTML).toBe("");
  });

  it("renders banner when offline", () => {
    Object.defineProperty(navigator, "onLine", { value: false, writable: true, configurable: true });
    render(<OfflineBanner />);
    expect(screen.getByText(/you are offline/i)).toBeInTheDocument();
  });

  it("shows banner when going offline via event", () => {
    Object.defineProperty(navigator, "onLine", { value: true, writable: true, configurable: true });
    render(<OfflineBanner />);

    // Initially should not show
    expect(screen.queryByText(/you are offline/i)).toBeNull();

    // Simulate going offline
    act(() => {
      window.dispatchEvent(new Event("offline"));
    });

    expect(screen.getByText(/you are offline/i)).toBeInTheDocument();
  });

  it("hides banner when coming back online", () => {
    Object.defineProperty(navigator, "onLine", { value: false, writable: true, configurable: true });
    render(<OfflineBanner />);
    expect(screen.getByText(/you are offline/i)).toBeInTheDocument();

    // Simulate coming back online
    act(() => {
      window.dispatchEvent(new Event("online"));
    });

    expect(screen.queryByText(/you are offline/i)).toBeNull();
  });

  it("banner has correct styling classes", () => {
    Object.defineProperty(navigator, "onLine", { value: false, writable: true, configurable: true });
    render(<OfflineBanner />);
    const banner = screen.getByText(/you are offline/i);
    expect(banner.className).toContain("fixed");
    expect(banner.className).toContain("z-50");
  });
});
