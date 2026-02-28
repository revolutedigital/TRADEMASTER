import { create } from "zustand";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface AuthState {
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  login: (username: string, password: string) => Promise<boolean>;
  logout: () => void;
  restoreSession: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  token: null,
  isAuthenticated: false,
  isLoading: false,
  error: null,

  login: async (username, password) => {
    set({ isLoading: true, error: null });
    try {
      const res = await fetch(`${API_URL}/api/v1/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({ detail: "Login failed" }));
        set({ isLoading: false, error: data.detail || "Invalid credentials" });
        return false;
      }

      const data = await res.json();
      const token = data.access_token;

      localStorage.setItem("tm_token", token);
      set({ token, isAuthenticated: true, isLoading: false, error: null });
      return true;
    } catch {
      set({ isLoading: false, error: "Connection failed" });
      return false;
    }
  },

  logout: () => {
    localStorage.removeItem("tm_token");
    set({ token: null, isAuthenticated: false, error: null });
  },

  restoreSession: () => {
    const token = localStorage.getItem("tm_token");
    if (token) {
      set({ token, isAuthenticated: true });
    }
  },
}));
