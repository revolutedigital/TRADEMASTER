"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Logo } from "./logo";
import {
  LayoutDashboard,
  CandlestickChart,
  Briefcase,
  Zap,
  FlaskConical,
  Settings,
  Shield,
  Menu,
  X,
  Brain,
  Bell,
  BarChart3,
  ChevronDown,
  History,
  FileText,
  GitCompare,
  PieChart,
  DollarSign,
  Wrench,
} from "lucide-react";

interface NavItem {
  href: string;
  label: string;
  icon: React.ElementType;
  children?: { href: string; label: string; icon: React.ElementType }[];
}

const navItems: NavItem[] = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  {
    href: "/trading",
    label: "Trading",
    icon: CandlestickChart,
    children: [
      { href: "/trading", label: "Terminal", icon: CandlestickChart },
      { href: "/trading/history", label: "History", icon: History },
      { href: "/trading/journal", label: "Journal", icon: FileText },
      { href: "/trading/strategy-builder", label: "Strategy", icon: Wrench },
    ],
  },
  {
    href: "/portfolio",
    label: "Portfolio",
    icon: Briefcase,
    children: [
      { href: "/portfolio", label: "Overview", icon: Briefcase },
      { href: "/portfolio/optimizer", label: "Optimizer", icon: PieChart },
      { href: "/portfolio/fees", label: "Fees", icon: DollarSign },
    ],
  },
  { href: "/signals", label: "Signals", icon: Zap },
  {
    href: "/backtest",
    label: "Backtest",
    icon: FlaskConical,
    children: [
      { href: "/backtest", label: "Run", icon: FlaskConical },
      { href: "/backtest/compare", label: "Compare", icon: GitCompare },
    ],
  },
  { href: "/ml", label: "ML/AI", icon: Brain },
  { href: "/sentiment", label: "Sentiment", icon: BarChart3 },
  { href: "/alerts", label: "Alerts", icon: Bell },
  { href: "/settings", label: "Settings", icon: Settings },
];

function NavLink({
  item,
  pathname,
  onNavigate,
}: {
  item: NavItem;
  pathname: string;
  onNavigate?: () => void;
}) {
  const isActive = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
  const [expanded, setExpanded] = useState(isActive && !!item.children);

  useEffect(() => {
    if (isActive && item.children) setExpanded(true);
  }, [isActive, item.children]);

  if (!item.children) {
    return (
      <Link
        href={item.href}
        onClick={onNavigate}
        aria-current={isActive ? "page" : undefined}
        className={cn(
          "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-all duration-150",
          isActive
            ? "bg-[var(--color-primary-light)] text-[var(--color-primary)]"
            : "text-[var(--color-text-muted)] hover:bg-[var(--color-surface-hover)] hover:text-[var(--color-text)]"
        )}
      >
        <item.icon className="h-4 w-4" />
        {item.label}
      </Link>
    );
  }

  return (
    <div>
      <button
        onClick={() => setExpanded(!expanded)}
        className={cn(
          "flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-all duration-150",
          isActive
            ? "text-[var(--color-primary)]"
            : "text-[var(--color-text-muted)] hover:bg-[var(--color-surface-hover)] hover:text-[var(--color-text)]"
        )}
      >
        <item.icon className="h-4 w-4" />
        <span className="flex-1 text-left">{item.label}</span>
        <ChevronDown
          className={cn("h-3.5 w-3.5 transition-transform duration-200", expanded && "rotate-180")}
        />
      </button>
      {expanded && (
        <div className="ml-4 mt-0.5 space-y-0.5 border-l border-[var(--color-border)] pl-3 animate-fade-in">
          {item.children.map((child) => {
            const childActive = pathname === child.href;
            return (
              <Link
                key={child.href}
                href={child.href}
                onClick={onNavigate}
                aria-current={childActive ? "page" : undefined}
                className={cn(
                  "flex items-center gap-2.5 rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors",
                  childActive
                    ? "text-[var(--color-primary)] bg-[var(--color-primary-light)]"
                    : "text-[var(--color-text-muted)] hover:text-[var(--color-text)]"
                )}
              >
                <child.icon className="h-3.5 w-3.5" />
                {child.label}
              </Link>
            );
          })}
        </div>
      )}
    </div>
  );
}

function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname();

  return (
    <>
      {/* Logo */}
      <div className="border-b border-[var(--color-border)] px-4 py-4">
        <Logo size="sm" />
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto space-y-0.5 p-3" role="navigation" aria-label="Main navigation">
        {navItems.map((item) => (
          <NavLink key={item.href} item={item} pathname={pathname} onNavigate={onNavigate} />
        ))}
      </nav>

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

  useEffect(() => {
    setOpen(false);
  }, [pathname]);

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="lg:hidden p-2 rounded-md text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface-hover)] transition-colors"
        aria-label="Open navigation menu"
      >
        <Menu className="h-5 w-5" />
      </button>

      {open && (
        <div
          className="fixed inset-0 z-40 bg-black/50 backdrop-blur-sm lg:hidden animate-fade-in"
          onClick={() => setOpen(false)}
          aria-hidden="true"
        />
      )}

      <div
        className={cn(
          "fixed inset-y-0 left-0 z-50 w-64 flex flex-col bg-[var(--color-surface)] border-r border-[var(--color-border)] transform transition-transform duration-200 ease-in-out lg:hidden",
          open ? "translate-x-0" : "-translate-x-full"
        )}
        role="dialog"
        aria-modal="true"
        aria-label="Navigation menu"
      >
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
