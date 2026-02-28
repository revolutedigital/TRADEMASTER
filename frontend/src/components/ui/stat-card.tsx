"use client";

import { Card } from "./card";
import { cn } from "@/lib/utils";

interface StatCardProps {
  label: string;
  value: string;
  change?: string;
  positive?: boolean;
  icon?: React.ReactNode;
}

export function StatCard({ label, value, change, positive, icon }: StatCardProps) {
  return (
    <Card className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <span className="text-xs text-[var(--color-text-muted)]">{label}</span>
        {icon && <span className="text-[var(--color-text-muted)]">{icon}</span>}
      </div>
      <span className="text-xl font-semibold tracking-tight">{value}</span>
      {change && (
        <span
          className={cn(
            "text-xs font-medium",
            positive === true && "text-green-400",
            positive === false && "text-red-400",
            positive === undefined && "text-[var(--color-text-muted)]"
          )}
        >
          {change}
        </span>
      )}
    </Card>
  );
}
