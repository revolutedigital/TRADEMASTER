"use client";

import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

interface AnimatedNumberProps {
  value: number;
  format?: (n: number) => string;
  duration?: number;
  className?: string;
}

export function AnimatedNumber({ value, format, duration = 400, className }: AnimatedNumberProps) {
  const [display, setDisplay] = useState(value);
  const prevRef = useRef(value);
  const rafRef = useRef<number>(0);
  const [flash, setFlash] = useState<"up" | "down" | null>(null);

  useEffect(() => {
    const prev = prevRef.current;
    if (prev === value) return;

    setFlash(value > prev ? "up" : "down");
    const timeout = setTimeout(() => setFlash(null), 600);

    const startTime = performance.now();
    const diff = value - prev;

    const animate = (time: number) => {
      const elapsed = time - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay(prev + diff * eased);

      if (progress < 1) {
        rafRef.current = requestAnimationFrame(animate);
      } else {
        prevRef.current = value;
      }
    };

    rafRef.current = requestAnimationFrame(animate);

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      clearTimeout(timeout);
    };
  }, [value, duration]);

  const formatted = format ? format(display) : display.toFixed(2);

  return (
    <span
      className={cn(
        "tabular-nums transition-colors duration-300",
        flash === "up" && "text-[var(--color-success)]",
        flash === "down" && "text-[var(--color-danger)]",
        className
      )}
    >
      {formatted}
    </span>
  );
}
