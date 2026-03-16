import { describe, it, expect, beforeEach } from "vitest";

describe("themeStore", () => {
  beforeEach(() => {
    // Clear persisted storage
    localStorage.clear();
  });

  it("initializes with dark theme", async () => {
    const { useThemeStore } = await import("@/stores/themeStore");
    const state = useThemeStore.getState();
    expect(state.theme).toBe("dark");
  });

  it("setTheme changes to light", async () => {
    const { useThemeStore } = await import("@/stores/themeStore");
    useThemeStore.getState().setTheme("light");
    expect(useThemeStore.getState().theme).toBe("light");
  });

  it("toggleTheme switches from dark to light", async () => {
    const { useThemeStore } = await import("@/stores/themeStore");
    // Ensure dark first
    useThemeStore.getState().setTheme("dark");
    useThemeStore.getState().toggleTheme();
    expect(useThemeStore.getState().theme).toBe("light");
  });

  it("toggleTheme switches from light to dark", async () => {
    const { useThemeStore } = await import("@/stores/themeStore");
    useThemeStore.getState().setTheme("light");
    useThemeStore.getState().toggleTheme();
    expect(useThemeStore.getState().theme).toBe("dark");
  });
});
