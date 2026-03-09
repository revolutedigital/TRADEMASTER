"use client";

import { useState, type ReactNode } from "react";
import { AlertCircle, CheckCircle, AlertTriangle, Info, X } from "lucide-react";
import { cn } from "@/lib/utils";

type AlertVariant = "info" | "success" | "warning" | "error";

interface AlertProps {
  variant?: AlertVariant;
  title?: string;
  children: ReactNode;
  dismissible?: boolean;
  className?: string;
}

const variantConfig: Record<AlertVariant, { icon: typeof Info; bg: string; border: string; text: string }> = {
  info: { icon: Info, bg: "bg-[var(--color-info-light)]", border: "border-[var(--color-info)]/20", text: "text-[var(--color-info)]" },
  success: { icon: CheckCircle, bg: "bg-[var(--color-success-light)]", border: "border-[var(--color-success)]/20", text: "text-[var(--color-success)]" },
  warning: { icon: AlertTriangle, bg: "bg-[var(--color-warning-light)]", border: "border-[var(--color-warning)]/20", text: "text-[var(--color-warning)]" },
  error: { icon: AlertCircle, bg: "bg-[var(--color-danger-light)]", border: "border-[var(--color-danger)]/20", text: "text-[var(--color-danger)]" },
};

export function Alert({ variant = "info", title, children, dismissible, className }: AlertProps) {
  const [visible, setVisible] = useState(true);
  if (!visible) return null;

  const config = variantConfig[variant];
  const Icon = config.icon;

  return (
    <div
      role={variant === "error" ? "alert" : "status"}
      className={cn(
        "flex gap-3 rounded-[var(--radius-md)] border p-3 text-sm animate-fade-in",
        config.bg,
        config.border,
        className
      )}
    >
      <Icon className={cn("h-5 w-5 shrink-0 mt-0.5", config.text)} />
      <div className="flex-1 min-w-0">
        {title && <p className="font-medium mb-0.5">{title}</p>}
        <div className="text-[var(--color-text-muted)]">{children}</div>
      </div>
      {dismissible && (
        <button
          onClick={() => setVisible(false)}
          className="shrink-0 rounded p-0.5 text-[var(--color-text-muted)] hover:text-[var(--color-text)] transition-colors"
          aria-label="Dismiss"
        >
          <X className="h-4 w-4" />
        </button>
      )}
    </div>
  );
}
