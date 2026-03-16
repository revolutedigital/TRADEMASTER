import { describe, it, expect, beforeEach } from "vitest";
import { OfflineStore } from "@/lib/offline";

describe("OfflineStore", () => {
  let store: OfflineStore;

  beforeEach(() => {
    store = new OfflineStore();
    localStorage.clear();
  });

  it("caches and retrieves portfolio data", () => {
    const data = { equity: 10000, positions: [] };
    store.cachePortfolio(data);
    expect(store.getCachedPortfolio()).toEqual(data);
  });

  it("caches and retrieves positions data", () => {
    const data = [{ symbol: "BTCUSDT", qty: 1 }];
    store.cachePositions(data);
    expect(store.getCachedPositions()).toEqual(data);
  });

  it("returns null for non-existent keys", () => {
    expect(store.getCachedPortfolio()).toBeNull();
    expect(store.getCachedPositions()).toBeNull();
  });

  it("returns null for expired cache entries", () => {
    // Manually write an expired entry
    const expired = { data: { x: 1 }, timestamp: Date.now() - 6 * 60 * 1000 };
    localStorage.setItem("tm_cache_portfolio", JSON.stringify(expired));
    expect(store.getCachedPortfolio()).toBeNull();
  });

  it("clears all cached entries", () => {
    store.cachePortfolio({ a: 1 });
    store.cachePositions([]);
    localStorage.setItem("other_key", "preserve");
    store.clearCache();
    expect(store.getCachedPortfolio()).toBeNull();
    expect(store.getCachedPositions()).toBeNull();
    expect(localStorage.getItem("other_key")).toBe("preserve");
  });

  it("reports online status", () => {
    expect(store.isOnline()).toBe(true);
  });
});
