"use client";

import { useEffect, useRef, useCallback, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";

interface DialogProps {
  open: boolean;
  onClose: () => void;
  title?: string;
  description?: string;
  size?: "sm" | "md" | "lg" | "xl";
  children: ReactNode;
  className?: string;
}

const sizeStyles = {
  sm: "max-w-sm",
  md: "max-w-md",
  lg: "max-w-lg",
  xl: "max-w-xl",
};

export function Dialog({ open, onClose, title, description, size = "md", children, className }: DialogProps) {
  const overlayRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const previousFocus = useRef<HTMLElement | null>(null);

  // Focus trap
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
        return;
      }
      if (e.key !== "Tab" || !contentRef.current) return;

      const focusable = contentRef.current.querySelectorAll<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );
      const first = focusable[0];
      const last = focusable[focusable.length - 1];

      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last?.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first?.focus();
      }
    },
    [onClose]
  );

  useEffect(() => {
    if (open) {
      previousFocus.current = document.activeElement as HTMLElement;
      document.addEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "hidden";
      // Focus first focusable element
      requestAnimationFrame(() => {
        const first = contentRef.current?.querySelector<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        first?.focus();
      });
    }
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "";
      previousFocus.current?.focus();
    };
  }, [open, handleKeyDown]);

  if (!open) return null;

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Overlay */}
      <div
        ref={overlayRef}
        className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-fade-in"
        onClick={onClose}
        aria-hidden="true"
      />
      {/* Content */}
      <div
        ref={contentRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={title ? "dialog-title" : undefined}
        aria-describedby={description ? "dialog-desc" : undefined}
        className={cn(
          "relative w-full rounded-[var(--radius-xl)] border border-[var(--color-border)] bg-[var(--color-surface)] p-6 shadow-xl",
          "animate-scale-in",
          sizeStyles[size],
          className
        )}
      >
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 rounded-[var(--radius-sm)] p-1 text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface-hover)] transition-colors"
          aria-label="Close dialog"
        >
          <X className="h-4 w-4" />
        </button>

        {title && (
          <h2 id="dialog-title" className="text-lg font-semibold mb-1">
            {title}
          </h2>
        )}
        {description && (
          <p id="dialog-desc" className="text-sm text-[var(--color-text-muted)] mb-4">
            {description}
          </p>
        )}

        {children}
      </div>
    </div>,
    document.body
  );
}

interface DialogFooterProps {
  children: ReactNode;
  className?: string;
}

export function DialogFooter({ children, className }: DialogFooterProps) {
  return (
    <div className={cn("mt-6 flex items-center justify-end gap-3", className)}>
      {children}
    </div>
  );
}
