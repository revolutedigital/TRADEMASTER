"use client";

import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

interface LivePriceProps {
  price: number;
  decimals?: number;
  className?: string;
}

/**
 * Displays a price that flashes green/red on each tick, like Binance.
 */
export function LivePrice({ price, decimals = 2, className }: LivePriceProps) {
  const prevRef = useRef(price);
  const [flash, setFlash] = useState<"up" | "down" | null>(null);

  useEffect(() => {
    if (price === prevRef.current) return;

    setFlash(price > prevRef.current ? "up" : "down");
    prevRef.current = price;

    const t = setTimeout(() => setFlash(null), 300);
    return () => clearTimeout(t);
  }, [price]);

  const formatted = new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(price);

  return (
    <span
      className={cn(
        "font-mono transition-colors duration-150",
        flash === "up" && "text-green-400",
        flash === "down" && "text-red-400",
        !flash && "text-[var(--color-text)]",
        className,
      )}
    >
      {formatted}
    </span>
  );
}
