"use client";

import { Card } from "./card";
import { cn } from "@/lib/utils";

interface StatCardProps {
  label: string;
  value: string;
  change?: string;
  positive?: boolean;
  icon?: React.ReactNode;
  className?: string;
}

export function StatCard({ label, value, change, positive, icon, className }: StatCardProps) {
  return (
    <Card className={cn("flex flex-col gap-1.5 interactive", className)}>
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-[var(--color-text-muted)] uppercase tracking-wider">{label}</span>
        {icon && (
          <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-[var(--color-primary-light)] text-[var(--color-primary)]">
            {icon}
          </span>
        )}
      </div>
      <span className="text-2xl font-semibold tracking-tight tabular-nums">{value}</span>
      {change && (
        <span
          className={cn(
            "text-xs font-medium tabular-nums",
            positive === true && "text-[var(--color-success)]",
            positive === false && "text-[var(--color-danger)]",
            positive === undefined && "text-[var(--color-text-faint)]"
          )}
        >
          {change}
        </span>
      )}
    </Card>
  );
}
