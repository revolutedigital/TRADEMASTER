"use client";

import { cn } from "@/lib/utils";

interface SwitchProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: string;
  size?: "sm" | "md";
  disabled?: boolean;
  className?: string;
}

export function Switch({ checked, onChange, label, size = "md", disabled, className }: SwitchProps) {
  const sizes = {
    sm: { track: "h-5 w-9", thumb: "h-3.5 w-3.5", translate: "translate-x-4" },
    md: { track: "h-6 w-11", thumb: "h-4.5 w-4.5", translate: "translate-x-5" },
  };
  const s = sizes[size];

  return (
    <label className={cn("inline-flex items-center gap-2.5 cursor-pointer", disabled && "opacity-50 cursor-not-allowed", className)}>
      <button
        role="switch"
        type="button"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => !disabled && onChange(!checked)}
        className={cn(
          "relative inline-flex shrink-0 rounded-full transition-colors duration-200 ease-in-out focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-primary)] focus-visible:ring-offset-2",
          s.track,
          checked ? "bg-[var(--color-primary)]" : "bg-[var(--color-border)]"
        )}
      >
        <span
          className={cn(
            "pointer-events-none inline-block rounded-full bg-white shadow-sm transition-transform duration-200 ease-in-out",
            s.thumb,
            "translate-y-[3px] translate-x-[3px]",
            checked && s.translate
          )}
        />
      </button>
      {label && <span className="text-sm text-[var(--color-text)]">{label}</span>}
    </label>
  );
}
