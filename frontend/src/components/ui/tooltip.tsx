"use client";

import { useState, useRef, type ReactNode } from "react";
import { cn } from "@/lib/utils";

interface TooltipProps {
  content: string;
  side?: "top" | "bottom" | "left" | "right";
  children: ReactNode;
  className?: string;
}

export function Tooltip({ content, side = "top", children, className }: TooltipProps) {
  const [visible, setVisible] = useState(false);
  const timeoutRef = useRef<ReturnType<typeof setTimeout>>(undefined);
  const id = useRef(`tooltip-${Math.random().toString(36).slice(2, 8)}`).current;

  const show = () => {
    clearTimeout(timeoutRef.current);
    timeoutRef.current = setTimeout(() => setVisible(true), 200);
  };

  const hide = () => {
    clearTimeout(timeoutRef.current);
    timeoutRef.current = setTimeout(() => setVisible(false), 100);
  };

  const positions = {
    top: "bottom-full left-1/2 -translate-x-1/2 mb-2",
    bottom: "top-full left-1/2 -translate-x-1/2 mt-2",
    left: "right-full top-1/2 -translate-y-1/2 mr-2",
    right: "left-full top-1/2 -translate-y-1/2 ml-2",
  };

  return (
    <div
      className="relative inline-flex"
      onMouseEnter={show}
      onMouseLeave={hide}
      onFocus={show}
      onBlur={hide}
    >
      <div aria-describedby={visible ? id : undefined}>{children}</div>
      {visible && (
        <div
          id={id}
          role="tooltip"
          className={cn(
            "absolute z-50 whitespace-nowrap rounded-[var(--radius-sm)] px-2.5 py-1.5 text-xs font-medium",
            "bg-[var(--color-text)] text-[var(--color-background)]",
            "shadow-lg animate-fade-in pointer-events-none",
            positions[side],
            className
          )}
        >
          {content}
        </div>
      )}
    </div>
  );
}
