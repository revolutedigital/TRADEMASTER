import { cn } from "@/lib/utils";

interface ProgressProps {
  value: number; // 0-100
  max?: number;
  variant?: "default" | "success" | "danger" | "warning" | "gradient";
  size?: "sm" | "md";
  showValue?: boolean;
  label?: string;
  className?: string;
}

const variantBg: Record<string, string> = {
  default: "bg-[var(--color-primary)]",
  success: "bg-[var(--color-success)]",
  danger: "bg-[var(--color-danger)]",
  warning: "bg-[var(--color-warning)]",
  gradient: "bg-gradient-to-r from-[var(--color-primary)] to-purple-500",
};

export function Progress({ value, max = 100, variant = "default", size = "md", showValue, label, className }: ProgressProps) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100));

  return (
    <div className={cn("w-full", className)}>
      {(label || showValue) && (
        <div className="mb-1.5 flex items-center justify-between text-xs">
          {label && <span className="text-[var(--color-text-muted)]">{label}</span>}
          {showValue && <span className="font-mono text-[var(--color-text-muted)]">{pct.toFixed(0)}%</span>}
        </div>
      )}
      <div
        className={cn(
          "w-full overflow-hidden rounded-full bg-[var(--color-background)]",
          size === "sm" ? "h-1.5" : "h-2.5"
        )}
        role="progressbar"
        aria-valuenow={value}
        aria-valuemin={0}
        aria-valuemax={max}
        aria-label={label}
      >
        <div
          className={cn(
            "h-full rounded-full transition-all duration-500 ease-out",
            variantBg[variant]
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

interface SpinnerProps {
  size?: "sm" | "md" | "lg";
  className?: string;
}

export function Spinner({ size = "md", className }: SpinnerProps) {
  const sizes = { sm: "h-4 w-4", md: "h-6 w-6", lg: "h-8 w-8" };
  return (
    <svg
      className={cn("animate-spin text-[var(--color-primary)]", sizes[size], className)}
      fill="none"
      viewBox="0 0 24 24"
      aria-label="Loading"
      role="status"
    >
      <circle className="opacity-20" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
      <path className="opacity-80" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  );
}
