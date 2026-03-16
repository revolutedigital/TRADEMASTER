import { describe, it, expect, vi, beforeEach } from "vitest";
import { PluginManager, type PluginManifest } from "@/lib/plugin-sdk";

const sampleManifest: PluginManifest = {
  id: "test-plugin",
  name: "Test Plugin",
  version: "1.0.0",
  author: "Test",
  description: "A test plugin",
  entryPoint: "./test-plugin.js",
  permissions: ["read:portfolio", "read:market"],
  widgets: [
    { id: "w1", name: "Widget 1", defaultSize: { width: 2, height: 1 }, placement: "dashboard" },
    { id: "w2", name: "Widget 2", defaultSize: { width: 1, height: 1 }, placement: "sidebar" },
  ],
};

describe("PluginManager", () => {
  let manager: PluginManager;

  beforeEach(() => {
    manager = new PluginManager();
  });

  it("registers a plugin", async () => {
    await manager.registerPlugin(sampleManifest);
    expect(manager.getRegisteredPlugins()).toHaveLength(1);
    expect(manager.getRegisteredPlugins()[0].id).toBe("test-plugin");
  });

  it("throws on duplicate registration", async () => {
    await manager.registerPlugin(sampleManifest);
    await expect(manager.registerPlugin(sampleManifest)).rejects.toThrow("already registered");
  });

  it("throws on loading unregistered plugin", async () => {
    await expect(manager.loadPlugin("nonexistent")).rejects.toThrow("not found");
  });

  it("unloads a plugin", async () => {
    await manager.registerPlugin(sampleManifest);
    manager.unloadPlugin("test-plugin");
    expect(manager.getRegisteredPlugins()).toHaveLength(0);
  });

  it("returns all widgets", async () => {
    await manager.registerPlugin(sampleManifest);
    expect(manager.getWidgets()).toHaveLength(2);
  });

  it("filters widgets by placement", async () => {
    await manager.registerPlugin(sampleManifest);
    expect(manager.getWidgets("dashboard")).toHaveLength(1);
    expect(manager.getWidgets("dashboard")[0].name).toBe("Widget 1");
    expect(manager.getWidgets("sidebar")).toHaveLength(1);
    expect(manager.getWidgets("trading")).toHaveLength(0);
  });

  it("returns empty arrays when no plugins registered", () => {
    expect(manager.getRegisteredPlugins()).toHaveLength(0);
    expect(manager.getWidgets()).toHaveLength(0);
  });
});

describe("PluginContext", () => {
  let manager: PluginManager;

  beforeEach(() => {
    manager = new PluginManager();
    vi.restoreAllMocks();
  });

  it("creates context with permission-gated getPortfolio", async () => {
    const ctx = manager.createContext(["read:portfolio"]);
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      json: async () => ({ equity: 10000 }),
    } as Response);
    const data = await ctx.getPortfolio();
    expect(data).toEqual({ equity: 10000 });
  });

  it("denies getPortfolio without permission", async () => {
    const ctx = manager.createContext([]);
    await expect(ctx.getPortfolio()).rejects.toThrow("Permission denied");
  });

  it("creates context with permission-gated getMarketData", async () => {
    const ctx = manager.createContext(["read:market"]);
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      json: async () => ({ price: 42000 }),
    } as Response);
    const data = await ctx.getMarketData("BTCUSDT");
    expect(data).toEqual({ price: 42000 });
  });

  it("denies getMarketData without permission", async () => {
    const ctx = manager.createContext([]);
    await expect(ctx.getMarketData("BTCUSDT")).rejects.toThrow("Permission denied");
  });

  it("creates context with permission-gated getTrades", async () => {
    const ctx = manager.createContext(["read:trades"]);
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      json: async () => ([{ id: 1 }]),
    } as Response);
    const data = await ctx.getTrades(10);
    expect(data).toEqual([{ id: 1 }]);
  });

  it("denies getTrades without permission", async () => {
    const ctx = manager.createContext([]);
    await expect(ctx.getTrades()).rejects.toThrow("Permission denied");
  });

  it("creates context with permission-gated createAlert", async () => {
    const ctx = manager.createContext(["write:alerts"]);
    vi.spyOn(globalThis, "fetch").mockResolvedValue({} as Response);
    await expect(ctx.createAlert("BTCUSDT", "above", 50000)).resolves.toBeUndefined();
  });

  it("denies createAlert without permission", async () => {
    const ctx = manager.createContext([]);
    await expect(ctx.createAlert("BTCUSDT", "above", 50000)).rejects.toThrow("Permission denied");
  });

  it("showNotification dispatches custom event", () => {
    const ctx = manager.createContext([]);
    const handler = vi.fn();
    window.addEventListener("plugin:notification", handler);
    ctx.showNotification("Hello", "success");
    expect(handler).toHaveBeenCalled();
    window.removeEventListener("plugin:notification", handler);
  });
});
