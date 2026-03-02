import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock fetch before importing store
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe("authStore", () => {
  beforeEach(() => {
    mockFetch.mockReset();
    localStorage.clear();
  });

  it("initializes with no token", async () => {
    const { useAuthStore } = await import("@/stores/authStore");
    const state = useAuthStore.getState();
    expect(state.isAuthenticated).toBe(false);
  });

  it("login sets authenticated state", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ access_token: "test-token" }),
    });

    const { useAuthStore } = await import("@/stores/authStore");
    await useAuthStore.getState().login("admin", "password");

    const state = useAuthStore.getState();
    expect(state.isAuthenticated).toBe(true);
  });

  it("logout clears state", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({}),
    });

    const { useAuthStore } = await import("@/stores/authStore");
    // First set authenticated
    useAuthStore.setState({ isAuthenticated: true, token: "test" });

    await useAuthStore.getState().logout();

    const state = useAuthStore.getState();
    expect(state.isAuthenticated).toBe(false);
    expect(state.token).toBeNull();
  });
});
