"use client";

import { useMarketStore } from "@/stores/marketStore";
import { useAuthStore } from "@/stores/authStore";
import { formatCurrency, formatPercent } from "@/lib/utils";
import { cn } from "@/lib/utils";
import { MobileSidebar } from "@/components/ui/sidebar";
import { NotificationBell } from "@/components/ui/notification-bell";
import { useThemeStore } from "@/stores/themeStore";
import { Sun, Moon } from "lucide-react";

export function Header() {
  const { prices } = useMarketStore();
  const logout = useAuthStore((s) => s.logout);
  const { theme, toggleTheme } = useThemeStore();

  const symbols = ["BTCUSDT", "ETHUSDT"];

  return (
    <header className="flex h-14 items-center justify-between border-b border-[var(--color-border)] bg-[var(--color-surface)] px-3 sm:px-6">
      {/* Left side: mobile menu + ticker strip */}
      <div className="flex items-center gap-2 sm:gap-6">
        <MobileSidebar />

        {symbols.map((symbol) => {
          const ticker = prices[symbol];
          const change = ticker?.change_24h ?? 0;
          const positive = change >= 0;

          return (
            <div key={symbol} className="flex items-center gap-1.5 sm:gap-3">
              <span className="text-xs sm:text-sm font-semibold">
                {symbol.replace("USDT", "")}
              </span>
              <span className="text-xs sm:text-sm font-mono">
                {ticker ? formatCurrency(ticker.price) : "---"}
              </span>
              <span
                className={cn(
                  "text-xs font-medium hidden sm:inline",
                  positive ? "text-green-400" : "text-red-400"
                )}
              >
                {ticker ? formatPercent(change) : "---"}
              </span>
            </div>
          );
        })}
      </div>

      {/* Right side */}
      <div className="flex items-center gap-2 sm:gap-4">
        <div className="hidden sm:flex items-center gap-1.5" aria-live="polite">
          <div className="h-2 w-2 rounded-full bg-green-400 animate-pulse-glow" />
          <span className="text-xs text-[var(--color-text-muted)]">Connected</span>
        </div>
        <NotificationBell />
        <button
          onClick={toggleTheme}
          className="p-1.5 rounded-md text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface-hover)] transition-colors"
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
        >
          {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </button>
        <button
          onClick={logout}
          className="text-xs text-[var(--color-text-muted)] transition-colors hover:text-[var(--color-danger)]"
          aria-label="Log out"
        >
          Logout
        </button>
      </div>
    </header>
  );
}
