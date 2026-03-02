/**
 * Offline data caching using localStorage for critical portfolio data.
 * Falls back gracefully when offline.
 */

const CACHE_PREFIX = "tm_cache_";
const CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes

interface CacheEntry<T> {
  data: T;
  timestamp: number;
}

export class OfflineStore {
  private set<T>(key: string, data: T): void {
    try {
      const entry: CacheEntry<T> = { data, timestamp: Date.now() };
      localStorage.setItem(`${CACHE_PREFIX}${key}`, JSON.stringify(entry));
    } catch {
      // Storage full or unavailable
    }
  }

  private get<T>(key: string): T | null {
    try {
      const raw = localStorage.getItem(`${CACHE_PREFIX}${key}`);
      if (!raw) return null;
      const entry: CacheEntry<T> = JSON.parse(raw);
      if (Date.now() - entry.timestamp > CACHE_TTL_MS) {
        localStorage.removeItem(`${CACHE_PREFIX}${key}`);
        return null;
      }
      return entry.data;
    } catch {
      return null;
    }
  }

  cachePortfolio(data: unknown): void {
    this.set("portfolio", data);
  }

  getCachedPortfolio<T>(): T | null {
    return this.get<T>("portfolio");
  }

  cachePositions(data: unknown): void {
    this.set("positions", data);
  }

  getCachedPositions<T>(): T | null {
    return this.get<T>("positions");
  }

  isOnline(): boolean {
    return typeof navigator !== "undefined" ? navigator.onLine : true;
  }

  clearCache(): void {
    if (typeof localStorage === "undefined") return;
    const keys = Object.keys(localStorage).filter((k) => k.startsWith(CACHE_PREFIX));
    keys.forEach((k) => localStorage.removeItem(k));
  }
}

export const offlineStore = new OfflineStore();
