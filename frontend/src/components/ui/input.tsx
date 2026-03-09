import { forwardRef } from "react";
import { cn } from "@/lib/utils";

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  helperText?: string;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ label, error, helperText, leftIcon, rightIcon, className, id, ...props }, ref) => {
    const inputId = id || label?.toLowerCase().replace(/\s+/g, "-");
    const errorId = error ? `${inputId}-error` : undefined;
    const helperId = helperText ? `${inputId}-helper` : undefined;

    return (
      <div className="space-y-1.5">
        {label && (
          <label
            htmlFor={inputId}
            className="block text-sm font-medium text-[var(--color-text-muted)]"
          >
            {label}
          </label>
        )}
        <div className="relative">
          {leftIcon && (
            <div className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--color-text-faint)]">
              {leftIcon}
            </div>
          )}
          <input
            ref={ref}
            id={inputId}
            className={cn(
              "w-full rounded-[var(--radius-md)] border bg-[var(--color-background)] px-3 py-2 text-sm text-[var(--color-text)]",
              "placeholder:text-[var(--color-text-faint)]",
              "transition-all duration-150",
              "focus:outline-none focus:ring-2 focus:ring-offset-0",
              error
                ? "border-[var(--color-danger)] focus:ring-[var(--color-danger)]/30"
                : "border-[var(--color-border)] focus:border-[var(--color-primary)] focus:ring-[var(--color-primary)]/30",
              leftIcon && "pl-10",
              rightIcon && "pr-10",
              "disabled:cursor-not-allowed disabled:opacity-50",
              className
            )}
            aria-invalid={error ? true : undefined}
            aria-describedby={
              [errorId, helperId].filter(Boolean).join(" ") || undefined
            }
            {...props}
          />
          {rightIcon && (
            <div className="absolute right-3 top-1/2 -translate-y-1/2 text-[var(--color-text-faint)]">
              {rightIcon}
            </div>
          )}
        </div>
        {error && (
          <p id={errorId} className="text-xs text-[var(--color-danger)]" role="alert">
            {error}
          </p>
        )}
        {helperText && !error && (
          <p id={helperId} className="text-xs text-[var(--color-text-faint)]">
            {helperText}
          </p>
        )}
      </div>
    );
  }
);

Input.displayName = "Input";
