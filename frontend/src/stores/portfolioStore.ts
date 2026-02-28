import { create } from "zustand";
import type {
  Position,
  Order,
  PortfolioSummary,
  RiskStatus,
  Signal,
} from "@/lib/types";
import { apiFetch } from "@/lib/utils";

interface PortfolioState {
  positions: Position[];
  orders: Order[];
  signals: Signal[];
  summary: PortfolioSummary | null;
  riskStatus: RiskStatus | null;

  fetchPositions: () => Promise<void>;
  fetchOrders: () => Promise<void>;
  fetchSignals: () => Promise<void>;
  fetchSummary: () => Promise<void>;
  fetchRiskStatus: () => Promise<void>;
  updatePosition: (position: Position) => void;
  updateRiskStatus: (status: RiskStatus) => void;
  addSignal: (signal: Signal) => void;
}

export const usePortfolioStore = create<PortfolioState>((set) => ({
  positions: [],
  orders: [],
  signals: [],
  summary: null,
  riskStatus: null,

  fetchPositions: async () => {
    try {
      const data = await apiFetch<Position[]>("/api/v1/portfolio/positions");
      set({ positions: data });
    } catch (err) {
      console.error("Failed to fetch positions:", err);
    }
  },

  fetchOrders: async () => {
    try {
      const data = await apiFetch<Order[]>("/api/v1/trading/orders");
      set({ orders: data });
    } catch (err) {
      console.error("Failed to fetch orders:", err);
    }
  },

  fetchSignals: async () => {
    try {
      const data = await apiFetch<Signal[]>("/api/v1/signals/history?limit=50");
      set({ signals: data });
    } catch (err) {
      console.error("Failed to fetch signals:", err);
    }
  },

  fetchSummary: async () => {
    try {
      const data = await apiFetch<PortfolioSummary>("/api/v1/portfolio/summary");
      set({ summary: data });
    } catch (err) {
      console.error("Failed to fetch summary:", err);
    }
  },

  fetchRiskStatus: async () => {
    try {
      const data = await apiFetch<RiskStatus>("/api/v1/portfolio/risk-status");
      set({ riskStatus: data });
    } catch (err) {
      console.error("Failed to fetch risk status:", err);
    }
  },

  updatePosition: (position) =>
    set((state) => ({
      positions: state.positions.map((p) =>
        p.id === position.id ? position : p
      ),
    })),

  updateRiskStatus: (status) => set({ riskStatus: status }),

  addSignal: (signal) =>
    set((state) => ({
      signals: [signal, ...state.signals].slice(0, 100),
    })),
}));
