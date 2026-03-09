"use client";

import { useState, useRef, useEffect, type ReactNode } from "react";
import { cn } from "@/lib/utils";

interface Tab {
  id: string;
  label: string;
  icon?: ReactNode;
}

interface TabsProps {
  tabs: Tab[];
  activeTab: string;
  onChange: (id: string) => void;
  variant?: "underline" | "pills";
  className?: string;
}

export function Tabs({ tabs, activeTab, onChange, variant = "underline", className }: TabsProps) {
  const tabsRef = useRef<HTMLDivElement>(null);
  const [indicatorStyle, setIndicatorStyle] = useState({ left: 0, width: 0 });

  useEffect(() => {
    if (variant !== "underline" || !tabsRef.current) return;
    const active = tabsRef.current.querySelector<HTMLButtonElement>(`[data-tab-id="${activeTab}"]`);
    if (active) {
      setIndicatorStyle({
        left: active.offsetLeft,
        width: active.offsetWidth,
      });
    }
  }, [activeTab, variant]);

  const handleKeyDown = (e: React.KeyboardEvent, index: number) => {
    let newIndex = index;
    if (e.key === "ArrowRight") newIndex = (index + 1) % tabs.length;
    else if (e.key === "ArrowLeft") newIndex = (index - 1 + tabs.length) % tabs.length;
    else return;

    e.preventDefault();
    onChange(tabs[newIndex].id);
    const btn = tabsRef.current?.querySelector<HTMLButtonElement>(`[data-tab-id="${tabs[newIndex].id}"]`);
    btn?.focus();
  };

  if (variant === "pills") {
    return (
      <div role="tablist" className={cn("flex gap-1 rounded-[var(--radius-lg)] bg-[var(--color-background)] p-1", className)}>
        {tabs.map((tab, i) => (
          <button
            key={tab.id}
            role="tab"
            aria-selected={activeTab === tab.id}
            tabIndex={activeTab === tab.id ? 0 : -1}
            onClick={() => onChange(tab.id)}
            onKeyDown={(e) => handleKeyDown(e, i)}
            className={cn(
              "flex items-center gap-1.5 rounded-[var(--radius-md)] px-3 py-1.5 text-sm font-medium transition-all duration-200",
              activeTab === tab.id
                ? "bg-[var(--color-surface)] text-[var(--color-text)] shadow-sm"
                : "text-[var(--color-text-muted)] hover:text-[var(--color-text)]"
            )}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>
    );
  }

  return (
    <div ref={tabsRef} role="tablist" className={cn("relative border-b border-[var(--color-border)]", className)}>
      <div className="flex gap-0">
        {tabs.map((tab, i) => (
          <button
            key={tab.id}
            role="tab"
            data-tab-id={tab.id}
            aria-selected={activeTab === tab.id}
            tabIndex={activeTab === tab.id ? 0 : -1}
            onClick={() => onChange(tab.id)}
            onKeyDown={(e) => handleKeyDown(e, i)}
            className={cn(
              "flex items-center gap-1.5 px-4 py-2.5 text-sm font-medium transition-colors",
              activeTab === tab.id
                ? "text-[var(--color-primary)]"
                : "text-[var(--color-text-muted)] hover:text-[var(--color-text)]"
            )}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>
      {/* Animated underline */}
      <div
        className="absolute bottom-0 h-0.5 bg-[var(--color-primary)] transition-all duration-200 ease-out rounded-full"
        style={{ left: indicatorStyle.left, width: indicatorStyle.width }}
      />
    </div>
  );
}

interface TabPanelProps {
  id: string;
  activeTab: string;
  children: ReactNode;
  className?: string;
}

export function TabPanel({ id, activeTab, children, className }: TabPanelProps) {
  if (id !== activeTab) return null;
  return (
    <div role="tabpanel" aria-labelledby={`tab-${id}`} className={cn("animate-fade-in", className)}>
      {children}
    </div>
  );
}
