"use client";

import { useState, useRef, useEffect, type ReactNode } from "react";
import { ChevronDown, Check } from "lucide-react";
import { cn } from "@/lib/utils";

interface SelectOption {
  value: string;
  label: string;
  icon?: ReactNode;
}

interface SelectProps {
  options: SelectOption[];
  value: string;
  onChange: (value: string) => void;
  label?: string;
  placeholder?: string;
  error?: string;
  disabled?: boolean;
  className?: string;
}

export function Select({ options, value, onChange, label, placeholder = "Select...", error, disabled, className }: SelectProps) {
  const [open, setOpen] = useState(false);
  const [focused, setFocused] = useState(-1);
  const ref = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLUListElement>(null);

  const selected = options.find((o) => o.value === value);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (disabled) return;
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      if (open && focused >= 0) {
        onChange(options[focused].value);
        setOpen(false);
      } else {
        setOpen(true);
      }
    } else if (e.key === "Escape") {
      setOpen(false);
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      if (!open) { setOpen(true); setFocused(0); return; }
      setFocused((f) => Math.min(f + 1, options.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setFocused((f) => Math.max(f - 1, 0));
    }
  };

  useEffect(() => {
    if (open && focused >= 0) {
      listRef.current?.children[focused]?.scrollIntoView({ block: "nearest" });
    }
  }, [focused, open]);

  return (
    <div ref={ref} className={cn("relative", className)}>
      {label && (
        <label className="mb-1.5 block text-sm font-medium text-[var(--color-text-muted)]">
          {label}
        </label>
      )}
      <button
        type="button"
        role="combobox"
        aria-expanded={open}
        aria-haspopup="listbox"
        disabled={disabled}
        onClick={() => !disabled && setOpen(!open)}
        onKeyDown={handleKeyDown}
        className={cn(
          "flex w-full items-center justify-between rounded-[var(--radius-md)] border px-3 py-2 text-sm",
          "transition-all duration-150",
          "focus:outline-none focus:ring-2 focus:ring-offset-0",
          error
            ? "border-[var(--color-danger)] focus:ring-[var(--color-danger)]/30"
            : "border-[var(--color-border)] focus:border-[var(--color-primary)] focus:ring-[var(--color-primary)]/30",
          "bg-[var(--color-background)]",
          disabled && "opacity-50 cursor-not-allowed"
        )}
      >
        <span className={cn(!selected && "text-[var(--color-text-faint)]")}>
          {selected ? (
            <span className="flex items-center gap-2">
              {selected.icon}
              {selected.label}
            </span>
          ) : (
            placeholder
          )}
        </span>
        <ChevronDown className={cn("h-4 w-4 text-[var(--color-text-faint)] transition-transform duration-200", open && "rotate-180")} />
      </button>

      {open && (
        <ul
          ref={listRef}
          role="listbox"
          className="absolute z-50 mt-1 w-full max-h-60 overflow-auto rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface)] py-1 shadow-lg animate-scale-in"
        >
          {options.map((option, i) => (
            <li
              key={option.value}
              role="option"
              aria-selected={option.value === value}
              className={cn(
                "flex items-center justify-between px-3 py-2 text-sm cursor-pointer transition-colors",
                i === focused && "bg-[var(--color-surface-hover)]",
                option.value === value && "text-[var(--color-primary)]"
              )}
              onClick={() => {
                onChange(option.value);
                setOpen(false);
              }}
              onMouseEnter={() => setFocused(i)}
            >
              <span className="flex items-center gap-2">
                {option.icon}
                {option.label}
              </span>
              {option.value === value && <Check className="h-4 w-4" />}
            </li>
          ))}
        </ul>
      )}

      {error && (
        <p className="mt-1 text-xs text-[var(--color-danger)]" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
