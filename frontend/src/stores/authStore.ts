import { create } from "zustand";

// All API calls use relative paths — Next.js rewrites /api/v1/* to backend

interface AuthState {
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  login: (username: string, password: string) => Promise<boolean>;
  logout: () => void;
  restoreSession: () => Promise<void>;
  refreshToken: () => Promise<boolean>;
}

export function getCsrfToken(): string | null {
  if (typeof document === "undefined") return null;
  const match = document.cookie.match(/csrf_token=([^;]+)/);
  return match ? match[1] : null;
}

export const useAuthStore = create<AuthState>((set) => ({
  token: null,
  isAuthenticated: false,
  isLoading: false,
  error: null,

  login: async (username, password) => {
    set({ isLoading: true, error: null });
    try {
      const res = await fetch("/api/v1/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
        credentials: "include",
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({ detail: "Login failed" }));
        set({ isLoading: false, error: data.detail || "Invalid credentials" });
        return false;
      }

      const data = await res.json();
      // Token kept in memory only; httpOnly cookies handle API auth
      // In-memory token used for WebSocket auth (WS can't send cookies)
      set({ token: data.access_token, isAuthenticated: true, isLoading: false, error: null });
      return true;
    } catch {
      set({ isLoading: false, error: "Connection failed" });
      return false;
    }
  },

  logout: async () => {
    try {
      await fetch("/api/v1/auth/logout", {
        method: "POST",
        credentials: "include",
      });
    } catch {
      // Ignore network errors on logout
    }
    set({ token: null, isAuthenticated: false, error: null });
  },

  restoreSession: async () => {
    // Try refreshing via httpOnly cookie to restore session on page load
    try {
      const res = await fetch("/api/v1/auth/refresh", {
        method: "POST",
        credentials: "include",
      });
      if (res.ok) {
        const data = await res.json();
        set({ token: data.access_token, isAuthenticated: true });
        return;
      }
    } catch {
      // No valid session
    }
    set({ token: null, isAuthenticated: false });
  },

  refreshToken: async () => {
    try {
      const res = await fetch("/api/v1/auth/refresh", {
        method: "POST",
        credentials: "include",
      });
      if (res.ok) {
        const data = await res.json();
        set({ token: data.access_token, isAuthenticated: true });
        return true;
      }
    } catch {
      // Ignore
    }
    return false;
  },
}));
