"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ChevronRight, Home } from "lucide-react";
import { cn } from "@/lib/utils";

const labelMap: Record<string, string> = {
  trading: "Trading",
  portfolio: "Portfolio",
  signals: "Signals",
  backtest: "Backtest",
  settings: "Settings",
  ml: "ML/AI",
  sentiment: "Sentiment",
  alerts: "Alerts",
  history: "History",
  journal: "Journal",
  "strategy-builder": "Strategy Builder",
  optimizer: "Optimizer",
  fees: "Fees",
  compare: "Compare",
};

export function Breadcrumbs({ className }: { className?: string }) {
  const pathname = usePathname();
  if (pathname === "/") return null;

  const segments = pathname.split("/").filter(Boolean);
  if (segments.length === 0) return null;

  const crumbs = segments.map((seg, i) => ({
    label: labelMap[seg] || seg.charAt(0).toUpperCase() + seg.slice(1),
    href: "/" + segments.slice(0, i + 1).join("/"),
    isLast: i === segments.length - 1,
  }));

  return (
    <nav aria-label="Breadcrumb" className={cn("flex items-center gap-1 text-xs text-[var(--color-text-faint)]", className)}>
      <Link href="/" className="hover:text-[var(--color-text-muted)] transition-colors" aria-label="Home">
        <Home className="h-3.5 w-3.5" />
      </Link>
      {crumbs.map((crumb) => (
        <span key={crumb.href} className="flex items-center gap-1">
          <ChevronRight className="h-3 w-3" />
          {crumb.isLast ? (
            <span className="text-[var(--color-text-muted)] font-medium" aria-current="page">
              {crumb.label}
            </span>
          ) : (
            <Link href={crumb.href} className="hover:text-[var(--color-text-muted)] transition-colors">
              {crumb.label}
            </Link>
          )}
        </span>
      ))}
    </nav>
  );
}
