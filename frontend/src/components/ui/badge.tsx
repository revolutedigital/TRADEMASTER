import { cn } from "@/lib/utils";

type Variant = "default" | "success" | "danger" | "warning" | "primary";

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: Variant;
  children: React.ReactNode;
}

const variantStyles: Record<Variant, string> = {
  default: "bg-[var(--color-border)] text-[var(--color-text)]",
  success: "bg-green-500/15 text-green-400",
  danger: "bg-red-500/15 text-red-400",
  warning: "bg-yellow-500/15 text-yellow-400",
  primary: "bg-indigo-500/15 text-indigo-400",
};

export function Badge({ variant = "default", className, children, ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium",
        variantStyles[variant],
        className
      )}
      {...props}
    >
      {children}
    </span>
  );
}
