import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatCurrency(value: number | undefined | null, decimals = 2): string {
  if (value == null || !isFinite(value)) return "$0.00";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
}

export function formatPercent(value: number | undefined | null, decimals = 2): string {
  if (value == null || !isFinite(value)) return "+0.00%";
  const sign = value >= 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(decimals)}%`;
}

export function formatNumber(value: number, decimals = 2): string {
  return new Intl.NumberFormat("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
}

export function formatCompactNumber(value: number): string {
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

export function timeAgo(date: Date | string): string {
  const now = new Date();
  const then = new Date(date);
  const seconds = Math.floor((now.getTime() - then.getTime()) / 1000);

  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function getCsrfToken(): string | null {
  if (typeof document === "undefined") return null;
  const match = document.cookie.match(/csrf_token=([^;]+)/);
  return match ? match[1] : null;
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  // Add cache-busting param for GET requests to prevent Next.js/browser caching
  const separator = path.includes("?") ? "&" : "?";
  const url = init?.method && init.method !== "GET"
    ? `${API_URL}${path}`
    : `${API_URL}${path}${separator}_t=${Date.now()}`;

  // Include CSRF token for mutating requests
  const isMutating = init?.method && ["POST", "PUT", "DELETE", "PATCH"].includes(init.method);
  const csrfToken = isMutating ? getCsrfToken() : null;

  // Send Bearer token as fallback (cross-origin cookies may be blocked)
  const { useAuthStore } = await import("@/stores/authStore");
  const token = useAuthStore.getState().token;

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...(csrfToken ? { "X-CSRF-Token": csrfToken } : {}),
    ...(init?.headers as Record<string, string> ?? {}),
  };

  const res = await fetch(url, {
    ...init,
    cache: "no-store",
    credentials: "include",
    headers,
  });

  // Handle authentication errors - try refresh before giving up
  if (res.status === 401) {
    if (typeof window !== "undefined") {
      const refreshRes = await fetch(`${API_URL}/api/v1/auth/refresh`, {
        method: "POST",
        credentials: "include",
      }).catch(() => null);

      if (refreshRes?.ok) {
        const refreshData = await refreshRes.json().catch(() => null);
        // Update in-memory token from refresh response
        if (refreshData?.access_token) {
          useAuthStore.getState().token = refreshData.access_token;
        }
        // Retry original request with new token
        const newToken = useAuthStore.getState().token;
        const retryRes = await fetch(url, {
          ...init,
          cache: "no-store",
          credentials: "include",
          headers: {
            "Content-Type": "application/json",
            ...(newToken ? { Authorization: `Bearer ${newToken}` } : {}),
            ...(csrfToken ? { "X-CSRF-Token": csrfToken } : {}),
            ...(init?.headers as Record<string, string> ?? {}),
          },
        });
        if (retryRes.ok) return retryRes.json();
      }

      window.location.reload();
    }
    throw new Error("Session expired");
  }

  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }

  return res.json();
}
