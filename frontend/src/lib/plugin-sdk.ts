/**
 * TradeMaster Plugin SDK
 * Enables third-party developers to create custom widgets and extensions.
 * Uses module federation pattern for runtime loading.
 */

export interface PluginManifest {
  id: string;
  name: string;
  version: string;
  author: string;
  description: string;
  entryPoint: string;
  permissions: PluginPermission[];
  widgets: WidgetDefinition[];
}

export type PluginPermission = "read:portfolio" | "read:market" | "read:trades" | "write:alerts" | "write:orders";

export interface WidgetDefinition {
  id: string;
  name: string;
  defaultSize: { width: number; height: number };
  minSize?: { width: number; height: number };
  placement: "dashboard" | "sidebar" | "trading" | "settings";
}

export interface PluginContext {
  getPortfolio: () => Promise<Record<string, unknown>>;
  getMarketData: (symbol: string) => Promise<Record<string, unknown>>;
  getTrades: (limit?: number) => Promise<Record<string, unknown>[]>;
  createAlert: (symbol: string, condition: string, price: number) => Promise<void>;
  onMarketUpdate: (symbol: string, callback: (data: unknown) => void) => () => void;
  showNotification: (message: string, type: "success" | "error" | "warning") => void;
}

export class PluginManager {
  private plugins: Map<string, PluginManifest> = new Map();
  private loadedModules: Map<string, unknown> = new Map();

  async registerPlugin(manifest: PluginManifest): Promise<void> {
    if (this.plugins.has(manifest.id)) {
      throw new Error(`Plugin ${manifest.id} already registered`);
    }
    this.plugins.set(manifest.id, manifest);
  }

  async loadPlugin(pluginId: string): Promise<unknown> {
    const manifest = this.plugins.get(pluginId);
    if (!manifest) throw new Error(`Plugin ${pluginId} not found`);
    
    if (this.loadedModules.has(pluginId)) {
      return this.loadedModules.get(pluginId);
    }

    try {
      const module = await import(/* webpackIgnore: true */ manifest.entryPoint);
      this.loadedModules.set(pluginId, module);
      return module;
    } catch (error) {
      console.error(`Failed to load plugin ${pluginId}:`, error);
      throw error;
    }
  }

  unloadPlugin(pluginId: string): void {
    this.loadedModules.delete(pluginId);
    this.plugins.delete(pluginId);
  }

  getRegisteredPlugins(): PluginManifest[] {
    return Array.from(this.plugins.values());
  }

  getWidgets(placement?: string): WidgetDefinition[] {
    const widgets: WidgetDefinition[] = [];
    for (const manifest of this.plugins.values()) {
      for (const widget of manifest.widgets) {
        if (!placement || widget.placement === placement) {
          widgets.push(widget);
        }
      }
    }
    return widgets;
  }

  createContext(permissions: PluginPermission[]): PluginContext {
    return {
      getPortfolio: async () => {
        if (!permissions.includes("read:portfolio")) throw new Error("Permission denied: read:portfolio");
        const res = await fetch("/api/v1/portfolio/summary", { credentials: "include" });
        return res.json();
      },
      getMarketData: async (symbol: string) => {
        if (!permissions.includes("read:market")) throw new Error("Permission denied: read:market");
        const res = await fetch(`/api/v1/market/tickers?symbol=${symbol}`, { credentials: "include" });
        return res.json();
      },
      getTrades: async (limit = 50) => {
        if (!permissions.includes("read:trades")) throw new Error("Permission denied: read:trades");
        const res = await fetch(`/api/v1/trading/history?limit=${limit}`, { credentials: "include" });
        return res.json();
      },
      createAlert: async (symbol, condition, price) => {
        if (!permissions.includes("write:alerts")) throw new Error("Permission denied: write:alerts");
        await fetch("/api/v1/alerts", {
          method: "POST", credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ symbol, condition, target_price: price }),
        });
      },
      onMarketUpdate: (symbol, callback) => {
        const ws = new WebSocket(`${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws/market`);
        ws.onmessage = (e) => {
          const data = JSON.parse(e.data);
          if (data.symbol === symbol) callback(data);
        };
        return () => ws.close();
      },
      showNotification: (message, type) => {
        const event = new CustomEvent("plugin:notification", { detail: { message, type } });
        window.dispatchEvent(event);
      },
    };
  }
}

export const pluginManager = new PluginManager();
