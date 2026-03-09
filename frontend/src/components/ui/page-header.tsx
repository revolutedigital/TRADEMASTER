import { type ReactNode } from "react";
import { Breadcrumbs } from "./breadcrumbs";
import { cn } from "@/lib/utils";

interface PageHeaderProps {
  title: string;
  description?: string;
  actions?: ReactNode;
  className?: string;
}

export function PageHeader({ title, description, actions, className }: PageHeaderProps) {
  return (
    <div className={cn("space-y-1 mb-6", className)}>
      <Breadcrumbs className="mb-3" />
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
          {description && (
            <p className="text-sm text-[var(--color-text-muted)] mt-0.5">{description}</p>
          )}
        </div>
        {actions && <div className="flex items-center gap-2">{actions}</div>}
      </div>
    </div>
  );
}
