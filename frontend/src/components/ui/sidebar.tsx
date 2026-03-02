"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  LayoutDashboard,
  CandlestickChart,
  Briefcase,
  Zap,
  FlaskConical,
  Settings,
  Shield,
  Activity,
  Menu,
  X,
  FileText,
} from "lucide-react";

const navItems = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/trading", label: "Trading", icon: CandlestickChart },
  { href: "/portfolio", label: "Portfolio", icon: Briefcase },
  { href: "/signals", label: "Signals", icon: Zap },
  { href: "/backtest", label: "Backtest", icon: FlaskConical },
  { href: "/settings", label: "Settings", icon: Settings },
];

const externalLinks = [
  { href: "/api/docs", label: "API Docs", icon: FileText },
];

function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname();

  return (
    <>
      {/* Logo */}
      <div className="flex items-center gap-2 border-b border-[var(--color-border)] px-4 py-4">
        <Activity className="h-6 w-6 text-[var(--color-primary)]" />
        <span className="text-lg font-bold tracking-tight">TradeMaster</span>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-0.5 p-3" role="navigation" aria-label="Main navigation">
        {navItems.map(({ href, label, icon: Icon }) => {
          const isActive = href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              onClick={onNavigate}
              aria-current={isActive ? "page" : undefined}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                isActive
                  ? "bg-[var(--color-primary)]/10 text-[var(--color-primary)]"
                  : "text-[var(--color-text-muted)] hover:bg-[var(--color-surface-hover)] hover:text-[var(--color-text)]"
              )}
            >
              <Icon className="h-4 w-4" />
              {label}
            </Link>
          );
        })}
      </nav>

      {/* External links */}
      <div className="border-t border-[var(--color-border)] px-3 pt-2 pb-1">
        {externalLinks.map(({ href, label, icon: Icon }) => (
          <a
            key={href}
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium text-[var(--color-text-muted)] hover:bg-[var(--color-surface-hover)] hover:text-[var(--color-text)] transition-colors"
          >
            <Icon className="h-4 w-4" />
            {label}
          </a>
        ))}
      </div>

      {/* Status footer */}
      <div className="border-t border-[var(--color-border)] p-3">
        <div className="flex items-center gap-2 rounded-lg bg-[var(--color-background)] px-3 py-2">
          <Shield className="h-4 w-4 text-green-400" />
          <span className="text-xs text-[var(--color-text-muted)]">Testnet Mode</span>
        </div>
      </div>
    </>
  );
}

export function Sidebar() {
  return (
    <aside className="hidden lg:flex h-screen w-56 flex-col border-r border-[var(--color-border)] bg-[var(--color-surface)]">
      <SidebarContent />
    </aside>
  );
}

export function MobileSidebar() {
  const [open, setOpen] = useState(false);
  const pathname = usePathname();

  // Close on route change
  useEffect(() => {
    setOpen(false);
  }, [pathname]);

  return (
    <>
      {/* Hamburger button */}
      <button
        onClick={() => setOpen(true)}
        className="lg:hidden p-2 rounded-md text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface-hover)] transition-colors"
        aria-label="Open navigation menu"
      >
        <Menu className="h-5 w-5" />
      </button>

      {/* Overlay */}
      {open && (
        <div
          className="fixed inset-0 z-40 bg-black/50 lg:hidden"
          onClick={() => setOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Drawer */}
      <div
        className={cn(
          "fixed inset-y-0 left-0 z-50 w-64 flex flex-col bg-[var(--color-surface)] border-r border-[var(--color-border)] transform transition-transform duration-200 ease-in-out lg:hidden",
          open ? "translate-x-0" : "-translate-x-full"
        )}
        role="dialog"
        aria-modal="true"
        aria-label="Navigation menu"
      >
        {/* Close button */}
        <div className="absolute top-3 right-3">
          <button
            onClick={() => setOpen(false)}
            className="p-1 rounded-md text-[var(--color-text-muted)] hover:text-[var(--color-text)]"
            aria-label="Close navigation menu"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <SidebarContent onNavigate={() => setOpen(false)} />
      </div>
    </>
  );
}
