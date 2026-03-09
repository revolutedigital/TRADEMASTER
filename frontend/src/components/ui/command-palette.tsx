"use client";

import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { createPortal } from "react-dom";
import { useRouter } from "next/navigation";
import {
  Search,
  LayoutDashboard,
  CandlestickChart,
  Briefcase,
  Zap,
  FlaskConical,
  Settings,
  ArrowUpCircle,
  ArrowDownCircle,
  Brain,
  Bell,
  BarChart3,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface CommandItem {
  id: string;
  label: string;
  group: string;
  icon: React.ReactNode;
  action: () => void;
  keywords?: string[];
}

export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();

  const navigate = useCallback(
    (path: string) => {
      router.push(path);
      setOpen(false);
    },
    [router]
  );

  const commands: CommandItem[] = useMemo(
    () => [
      { id: "dash", label: "Dashboard", group: "Navigation", icon: <LayoutDashboard className="h-4 w-4" />, action: () => navigate("/"), keywords: ["home"] },
      { id: "trade", label: "Trading Terminal", group: "Navigation", icon: <CandlestickChart className="h-4 w-4" />, action: () => navigate("/trading"), keywords: ["buy", "sell"] },
      { id: "port", label: "Portfolio", group: "Navigation", icon: <Briefcase className="h-4 w-4" />, action: () => navigate("/portfolio"), keywords: ["positions", "balance"] },
      { id: "sig", label: "AI Signals", group: "Navigation", icon: <Zap className="h-4 w-4" />, action: () => navigate("/signals"), keywords: ["predictions"] },
      { id: "bt", label: "Backtesting", group: "Navigation", icon: <FlaskConical className="h-4 w-4" />, action: () => navigate("/backtest"), keywords: ["simulate"] },
      { id: "ml", label: "ML/AI Dashboard", group: "Navigation", icon: <Brain className="h-4 w-4" />, action: () => navigate("/ml"), keywords: ["model", "ai"] },
      { id: "alert", label: "Price Alerts", group: "Navigation", icon: <Bell className="h-4 w-4" />, action: () => navigate("/alerts"), keywords: ["notify"] },
      { id: "sent", label: "Market Sentiment", group: "Navigation", icon: <BarChart3 className="h-4 w-4" />, action: () => navigate("/sentiment"), keywords: ["fear", "greed"] },
      { id: "set", label: "Settings", group: "Navigation", icon: <Settings className="h-4 w-4" />, action: () => navigate("/settings"), keywords: ["config"] },
      { id: "buy-btc", label: "Buy BTC", group: "Trading", icon: <ArrowUpCircle className="h-4 w-4 text-[var(--color-buy)]" />, action: () => navigate("/trading?action=buy&symbol=BTCUSDT"), keywords: ["long", "bitcoin"] },
      { id: "sell-btc", label: "Sell BTC", group: "Trading", icon: <ArrowDownCircle className="h-4 w-4 text-[var(--color-sell)]" />, action: () => navigate("/trading?action=sell&symbol=BTCUSDT"), keywords: ["short", "bitcoin"] },
      { id: "buy-eth", label: "Buy ETH", group: "Trading", icon: <ArrowUpCircle className="h-4 w-4 text-[var(--color-buy)]" />, action: () => navigate("/trading?action=buy&symbol=ETHUSDT"), keywords: ["long", "ethereum"] },
      { id: "sell-eth", label: "Sell ETH", group: "Trading", icon: <ArrowDownCircle className="h-4 w-4 text-[var(--color-sell)]" />, action: () => navigate("/trading?action=sell&symbol=ETHUSDT"), keywords: ["short", "ethereum"] },
      { id: "hist", label: "Trade History", group: "Navigation", icon: <CandlestickChart className="h-4 w-4" />, action: () => navigate("/trading/history"), keywords: ["past", "trades"] },
      { id: "journ", label: "Trade Journal", group: "Navigation", icon: <CandlestickChart className="h-4 w-4" />, action: () => navigate("/trading/journal"), keywords: ["notes"] },
    ],
    [navigate]
  );

  const filtered = useMemo(() => {
    if (!query) return commands;
    const q = query.toLowerCase();
    return commands.filter(
      (c) =>
        c.label.toLowerCase().includes(q) ||
        c.group.toLowerCase().includes(q) ||
        c.keywords?.some((k) => k.includes(q))
    );
  }, [commands, query]);

  const groups = useMemo(() => {
    const map = new Map<string, CommandItem[]>();
    filtered.forEach((c) => {
      if (!map.has(c.group)) map.set(c.group, []);
      map.get(c.group)!.push(c);
    });
    return map;
  }, [filtered]);

  // Open with Ctrl+K / Cmd+K
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setOpen((o) => !o);
        setQuery("");
        setSelected(0);
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, []);

  // Focus input when opened
  useEffect(() => {
    if (open) {
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") {
      setOpen(false);
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelected((s) => Math.min(s + 1, filtered.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelected((s) => Math.max(s - 1, 0));
    } else if (e.key === "Enter" && filtered[selected]) {
      filtered[selected].action();
      setOpen(false);
    }
  };

  if (!open) return null;

  let flatIndex = -1;

  return createPortal(
    <div className="fixed inset-0 z-[100] flex items-start justify-center pt-[20vh]">
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm animate-fade-in" onClick={() => setOpen(false)} />
      <div
        className="relative w-full max-w-lg rounded-[var(--radius-xl)] border border-[var(--color-border)] bg-[var(--color-surface)] shadow-2xl animate-scale-in overflow-hidden"
        role="dialog"
        aria-label="Command palette"
      >
        {/* Search input */}
        <div className="flex items-center gap-3 border-b border-[var(--color-border)] px-4">
          <Search className="h-4 w-4 text-[var(--color-text-muted)]" />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setSelected(0);
            }}
            onKeyDown={handleKeyDown}
            placeholder="Type a command or search..."
            className="flex-1 bg-transparent py-3.5 text-sm text-[var(--color-text)] placeholder:text-[var(--color-text-faint)] focus:outline-none"
          />
          <kbd className="hidden sm:inline-flex rounded border border-[var(--color-border)] px-1.5 py-0.5 text-[10px] text-[var(--color-text-faint)]">
            ESC
          </kbd>
        </div>

        {/* Results */}
        <div className="max-h-80 overflow-y-auto py-2">
          {filtered.length === 0 && (
            <p className="py-8 text-center text-sm text-[var(--color-text-muted)]">No results found</p>
          )}
          {Array.from(groups.entries()).map(([group, items]) => (
            <div key={group}>
              <div className="px-4 py-1.5 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-faint)]">
                {group}
              </div>
              {items.map((item) => {
                flatIndex++;
                const idx = flatIndex;
                return (
                  <button
                    key={item.id}
                    onClick={() => {
                      item.action();
                      setOpen(false);
                    }}
                    onMouseEnter={() => setSelected(idx)}
                    className={cn(
                      "flex w-full items-center gap-3 px-4 py-2 text-sm transition-colors",
                      idx === selected
                        ? "bg-[var(--color-primary-light)] text-[var(--color-primary)]"
                        : "text-[var(--color-text)] hover:bg-[var(--color-surface-hover)]"
                    )}
                  >
                    {item.icon}
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between border-t border-[var(--color-border)] px-4 py-2 text-[10px] text-[var(--color-text-faint)]">
          <span>Navigate with <kbd className="font-mono">↑↓</kbd> · Select with <kbd className="font-mono">↵</kbd></span>
          <span>
            <kbd className="font-mono">⌘K</kbd> to toggle
          </span>
        </div>
      </div>
    </div>,
    document.body
  );
}
