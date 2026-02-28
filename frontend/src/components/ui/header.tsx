"use client";

import { useMarketStore } from "@/stores/marketStore";
import { useAuthStore } from "@/stores/authStore";
import { formatCurrency, formatPercent } from "@/lib/utils";
import { cn } from "@/lib/utils";

export function Header() {
  const { prices } = useMarketStore();
  const logout = useAuthStore((s) => s.logout);

  const symbols = ["BTCUSDT", "ETHUSDT"];

  return (
    <header className="flex h-14 items-center justify-between border-b border-[var(--color-border)] bg-[var(--color-surface)] px-6">
      {/* Ticker strip */}
      <div className="flex items-center gap-6">
        {symbols.map((symbol) => {
          const ticker = prices[symbol];
          const change = ticker?.change_24h ?? 0;
          const positive = change >= 0;

          return (
            <div key={symbol} className="flex items-center gap-3">
              <span className="text-sm font-semibold">
                {symbol.replace("USDT", "")}
              </span>
              <span className="text-sm font-mono">
                {ticker ? formatCurrency(ticker.price) : "---"}
              </span>
              <span
                className={cn(
                  "text-xs font-medium",
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
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-1.5">
          <div className="h-2 w-2 rounded-full bg-green-400 animate-pulse-glow" />
          <span className="text-xs text-[var(--color-text-muted)]">Connected</span>
        </div>
        <button
          onClick={logout}
          className="text-xs text-[var(--color-text-muted)] transition-colors hover:text-[var(--color-danger)]"
        >
          Logout
        </button>
      </div>
    </header>
  );
}
