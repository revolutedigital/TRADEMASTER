import { cn } from "@/lib/utils";

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  variant?: "default" | "glass" | "gradient";
}

export function Card({ className, variant = "default", children, ...props }: CardProps) {
  return (
    <div
      className={cn(
        "rounded-xl p-4 transition-colors duration-150",
        variant === "default" && "border border-[var(--color-border)] bg-[var(--color-surface)]",
        variant === "glass" && "glass",
        variant === "gradient" && "gradient-border",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}

export function CardHeader({
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLDivElement> & { children: React.ReactNode }) {
  return (
    <div className={cn("mb-3 flex items-center justify-between", className)} {...props}>
      {children}
    </div>
  );
}

export function CardTitle({
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLHeadingElement> & { children: React.ReactNode }) {
  return (
    <h3 className={cn("text-sm font-medium text-[var(--color-text-muted)] uppercase tracking-wider", className)} {...props}>
      {children}
    </h3>
  );
}

export function CardContent({
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLDivElement> & { children: React.ReactNode }) {
  return (
    <div className={cn("", className)} {...props}>
      {children}
    </div>
  );
}
