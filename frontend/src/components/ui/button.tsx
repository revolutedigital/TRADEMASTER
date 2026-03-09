import { cn } from "@/lib/utils";

type Variant = "default" | "primary" | "danger" | "ghost" | "success";
type Size = "sm" | "md" | "lg";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
}

const variantStyles: Record<Variant, string> = {
  default:
    "border border-[var(--color-border)] bg-[var(--color-surface)] text-[var(--color-text)] hover:bg-[var(--color-surface-hover)] hover:border-[var(--color-border-hover)]",
  primary:
    "bg-[var(--color-primary)] text-white hover:bg-[var(--color-primary-hover)] shadow-sm hover:shadow-md",
  danger:
    "bg-[var(--color-danger)] text-white hover:bg-red-700 shadow-sm",
  success:
    "bg-[var(--color-success)] text-white hover:bg-green-700 shadow-sm",
  ghost:
    "text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface-hover)]",
};

const sizeStyles: Record<Size, string> = {
  sm: "h-7 px-2.5 text-xs gap-1.5",
  md: "h-9 px-4 text-sm gap-2",
  lg: "h-11 px-6 text-base gap-2",
};

export function Button({
  variant = "default",
  size = "md",
  className,
  disabled,
  children,
  ...props
}: ButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center rounded-lg font-medium",
        "transition-all duration-150",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-primary)] focus-visible:ring-offset-1",
        "disabled:pointer-events-none disabled:opacity-50",
        "active:scale-[0.97]",
        variantStyles[variant],
        sizeStyles[size],
        className
      )}
      disabled={disabled}
      {...props}
    >
      {children}
    </button>
  );
}
