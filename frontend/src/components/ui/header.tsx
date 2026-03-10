"use client";

import { useMarketStore } from "@/stores/marketStore";
import { useAuthStore } from "@/stores/authStore";
import { formatCurrency, formatPercent } from "@/lib/utils";
import { cn } from "@/lib/utils";
import { MobileSidebar } from "@/components/ui/sidebar";
import { NotificationBell } from "@/components/ui/notification-bell";
import { useThemeStore } from "@/stores/themeStore";
import { Sun, Moon, Search, LogOut } from "lucide-react";
import { Tooltip } from "@/components/ui/tooltip";

export function Header() {
  const { prices } = useMarketStore();
  const logout = useAuthStore((s) => s.logout);
  const { theme, toggleTheme } = useThemeStore();

  const symbols = ["BTCUSDT", "ETHUSDT"];

  return (
    <>
      {/* Animated gradient top bar */}
      <div className="gradient-bar" aria-hidden="true" />

      <header className="flex h-13 items-center justify-between border-b border-[var(--color-border)] bg-[var(--color-surface)] px-3 sm:px-6">
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
                <span className="text-xs sm:text-sm font-mono tabular-nums">
                  {ticker ? formatCurrency(ticker.price) : "---"}
                </span>
                <span
                  className={cn(
                    "text-xs font-medium hidden sm:inline tabular-nums",
                    positive ? "text-[var(--color-success)]" : "text-[var(--color-danger)]"
                  )}
                >
                  {ticker ? formatPercent(change) : "---"}
                </span>
              </div>
            );
          })}
        </div>

        {/* Right side */}
        <div className="flex items-center gap-1.5 sm:gap-2">
          {/* Connection status */}
          <div className="hidden sm:flex items-center gap-1.5 mr-2" aria-live="polite">
            <div className="h-2 w-2 rounded-full bg-[var(--color-success)] animate-pulse-glow" />
            <span className="text-xs text-[var(--color-text-faint)]">Ao Vivo</span>
          </div>

          {/* Command Palette trigger */}
          <Tooltip content="Buscar (⌘K)">
            <button
              onClick={() => document.dispatchEvent(new KeyboardEvent("keydown", { key: "k", metaKey: true }))}
              className="p-2 rounded-lg text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface-hover)] transition-colors"
              aria-label="Open command palette"
            >
              <Search className="h-4 w-4" />
            </button>
          </Tooltip>

          <NotificationBell />

          <Tooltip content={theme === "dark" ? "Modo claro" : "Modo escuro"}>
            <button
              onClick={toggleTheme}
              className="p-2 rounded-lg text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface-hover)] transition-colors"
              aria-label={theme === "dark" ? "Modo claro" : "Modo escuro"}
            >
              {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </button>
          </Tooltip>

          <Tooltip content="Sair">
            <button
              onClick={logout}
              className="p-2 rounded-lg text-[var(--color-text-muted)] hover:text-[var(--color-danger)] hover:bg-[var(--color-danger-light)] transition-colors"
              aria-label="Sair"
            >
              <LogOut className="h-4 w-4" />
            </button>
          </Tooltip>
        </div>
      </header>
    </>
  );
}
