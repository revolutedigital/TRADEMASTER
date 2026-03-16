import { describe, it, expect, vi } from "vitest";
import { render } from "@testing-library/react";
import { ServiceWorkerRegister } from "@/components/sw-register";

describe("ServiceWorkerRegister", () => {
  it("renders null (no visible UI)", () => {
    const { container } = render(<ServiceWorkerRegister />);
    expect(container.innerHTML).toBe("");
  });

  it("attempts to register service worker if supported", () => {
    const mockRegister = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "serviceWorker", {
      value: { register: mockRegister },
      writable: true,
      configurable: true,
    });
    render(<ServiceWorkerRegister />);
    expect(mockRegister).toHaveBeenCalledWith("/sw.js");
  });
});
