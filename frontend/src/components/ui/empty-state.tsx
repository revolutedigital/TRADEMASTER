import { type ReactNode } from "react";
import { cn } from "@/lib/utils";
import { Button } from "./button";

interface EmptyStateProps {
  icon?: ReactNode;
  title: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
  className?: string;
}

export function EmptyState({ icon, title, description, action, className }: EmptyStateProps) {
  return (
    <div className={cn("flex flex-col items-center justify-center py-12 px-4 text-center", className)}>
      {icon && (
        <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-[var(--color-primary-light)] text-[var(--color-primary)]">
          {icon}
        </div>
      )}
      <h3 className="text-sm font-semibold text-[var(--color-text)]">{title}</h3>
      {description && (
        <p className="mt-1 max-w-sm text-sm text-[var(--color-text-muted)]">{description}</p>
      )}
      {action && (
        <Button variant="primary" size="sm" className="mt-4" onClick={action.onClick}>
          {action.label}
        </Button>
      )}
    </div>
  );
}
