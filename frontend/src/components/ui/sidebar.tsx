"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Logo } from "./logo";
import { CandlestickChart, Briefcase, Settings, Shield, Menu, X, History } from "lucide-react";

interface NavItem {
  href: string;
  label: string;
  icon: React.ElementType;
}

const navItems: NavItem[] = [
  { href: "/operar", label: "Operar", icon: CandlestickChart },
  { href: "/portfolio", label: "Portfólio", icon: Briefcase },
  { href: "/trading/history", label: "Histórico", icon: History },
  { href: "/settings", label: "Configurações", icon: Settings },
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
          <span className="text-xs text-[var(--color-text-muted)]">Execução protegida</span>
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
        aria-label="Abrir menu de navegação"
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
            aria-label="Fechar menu de navegação"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <SidebarContent onNavigate={() => setOpen(false)} />
      </div>
    </>
  );
}
