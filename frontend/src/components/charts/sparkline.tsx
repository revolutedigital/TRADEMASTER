"use client";

import { useMemo } from "react";
import { cn } from "@/lib/utils";

interface SparklineProps {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
  showArea?: boolean;
  className?: string;
}

export function Sparkline({ data, width = 64, height = 24, color, showArea = true, className }: SparklineProps) {
  const pathData = useMemo(() => {
    if (data.length < 2) return { line: "", area: "" };

    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    const padding = 1;

    const points = data.map((val, i) => ({
      x: padding + (i / (data.length - 1)) * (width - padding * 2),
      y: padding + (1 - (val - min) / range) * (height - padding * 2),
    }));

    const line = points.map((p, i) => `${i === 0 ? "M" : "L"}${p.x},${p.y}`).join(" ");
    const area = `${line} L${points[points.length - 1].x},${height} L${points[0].x},${height} Z`;

    return { line, area };
  }, [data, width, height]);

  const isPositive = data.length >= 2 && data[data.length - 1] >= data[0];
  const strokeColor = color || (isPositive ? "var(--color-success)" : "var(--color-danger)");

  if (data.length < 2) return null;

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className={cn("inline-block", className)}
      aria-hidden="true"
    >
      {showArea && (
        <path d={pathData.area} fill={strokeColor} opacity={0.1} />
      )}
      <path
        d={pathData.line}
        fill="none"
        stroke={strokeColor}
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
