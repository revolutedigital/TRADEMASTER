import { describe, it, expect, vi, beforeEach } from "vitest";
import { useAuthStore, getCsrfToken } from "@/stores/authStore";

describe("getCsrfToken", () => {
  it("returns null when no csrf cookie", () => {
    Object.defineProperty(document, "cookie", { value: "", writable: true, configurable: true });
    expect(getCsrfToken()).toBeNull();
  });

  it("extracts csrf token from cookie", () => {
    Object.defineProperty(document, "cookie", {
      value: "other=foo; csrf_token=abc123; session=xyz",
      writable: true,
      configurable: true,
    });
    expect(getCsrfToken()).toBe("abc123");
  });
});

describe("useAuthStore", () => {
  beforeEach(() => {
    useAuthStore.setState({
      token: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,
    });
    vi.restoreAllMocks();
  });

  it("has correct initial state", () => {
    const state = useAuthStore.getState();
    expect(state.token).toBeNull();
    expect(state.isAuthenticated).toBe(false);
    expect(state.isLoading).toBe(false);
    expect(state.error).toBeNull();
  });

  it("login sets token and isAuthenticated on success", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: async () => ({ access_token: "test-token" }),
    } as Response);

    const result = await useAuthStore.getState().login("admin", "pass");
    expect(result).toBe(true);
    expect(useAuthStore.getState().token).toBe("test-token");
    expect(useAuthStore.getState().isAuthenticated).toBe(true);
  });

  it("login sets error on failure", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false,
      json: async () => ({ detail: "Invalid credentials" }),
    } as Response);

    const result = await useAuthStore.getState().login("admin", "wrong");
    expect(result).toBe(false);
    expect(useAuthStore.getState().error).toBe("Invalid credentials");
    expect(useAuthStore.getState().isAuthenticated).toBe(false);
  });

  it("login handles network error", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("Network"));

    const result = await useAuthStore.getState().login("admin", "pass");
    expect(result).toBe(false);
    expect(useAuthStore.getState().error).toBe("Connection failed");
  });

  it("logout clears state", async () => {
    useAuthStore.setState({ token: "tok", isAuthenticated: true });
    vi.spyOn(globalThis, "fetch").mockResolvedValue({ ok: true } as Response);

    await useAuthStore.getState().logout();
    expect(useAuthStore.getState().token).toBeNull();
    expect(useAuthStore.getState().isAuthenticated).toBe(false);
  });

  it("logout handles network error gracefully", async () => {
    useAuthStore.setState({ token: "tok", isAuthenticated: true });
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("Net fail"));

    await useAuthStore.getState().logout();
    expect(useAuthStore.getState().token).toBeNull();
    expect(useAuthStore.getState().isAuthenticated).toBe(false);
  });

  it("restoreSession sets token on successful refresh", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: async () => ({ access_token: "refreshed" }),
    } as Response);

    await useAuthStore.getState().restoreSession();
    expect(useAuthStore.getState().token).toBe("refreshed");
    expect(useAuthStore.getState().isAuthenticated).toBe(true);
  });

  it("restoreSession clears auth on failed refresh", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({ ok: false } as Response);

    await useAuthStore.getState().restoreSession();
    expect(useAuthStore.getState().token).toBeNull();
    expect(useAuthStore.getState().isAuthenticated).toBe(false);
  });

  it("refreshToken returns true on success", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: async () => ({ access_token: "new-tok" }),
    } as Response);

    const result = await useAuthStore.getState().refreshToken();
    expect(result).toBe(true);
    expect(useAuthStore.getState().token).toBe("new-tok");
  });

  it("refreshToken returns false on failure", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({ ok: false } as Response);

    const result = await useAuthStore.getState().refreshToken();
    expect(result).toBe(false);
  });
});
