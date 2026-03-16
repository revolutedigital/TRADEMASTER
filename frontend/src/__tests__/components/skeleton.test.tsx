import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import {
  Skeleton,
  CardSkeleton,
  StatCardSkeleton,
  TableSkeleton,
  ChartSkeleton,
  DashboardSkeleton,
} from "@/components/ui/skeleton";

describe("Skeleton", () => {
  it("renders aria-hidden div", () => {
    const { container } = render(<Skeleton />);
    expect(container.querySelector("[aria-hidden='true']")).toBeInTheDocument();
  });

  it("applies custom className", () => {
    const { container } = render(<Skeleton className="h-4 w-full" />);
    expect(container.querySelector("div")?.className).toContain("h-4");
  });
});

describe("CardSkeleton", () => {
  it("renders multiple skeleton elements", () => {
    const { container } = render(<CardSkeleton />);
    const skeletons = container.querySelectorAll("[aria-hidden='true']");
    expect(skeletons.length).toBe(3);
  });
});

describe("StatCardSkeleton", () => {
  it("renders skeleton elements", () => {
    const { container } = render(<StatCardSkeleton />);
    const skeletons = container.querySelectorAll("[aria-hidden='true']");
    expect(skeletons.length).toBeGreaterThanOrEqual(3);
  });
});

describe("TableSkeleton", () => {
  it("renders header + default 5 rows", () => {
    const { container } = render(<TableSkeleton />);
    const skeletons = container.querySelectorAll("[aria-hidden='true']");
    expect(skeletons.length).toBe(6); // 1 header + 5 rows
  });

  it("respects custom row count", () => {
    const { container } = render(<TableSkeleton rows={3} />);
    const skeletons = container.querySelectorAll("[aria-hidden='true']");
    expect(skeletons.length).toBe(4); // 1 header + 3 rows
  });
});

describe("ChartSkeleton", () => {
  it("renders skeleton elements", () => {
    const { container } = render(<ChartSkeleton />);
    const skeletons = container.querySelectorAll("[aria-hidden='true']");
    expect(skeletons.length).toBe(2);
  });
});

describe("DashboardSkeleton", () => {
  it("renders a combination of card, chart, and table skeletons", () => {
    const { container } = render(<DashboardSkeleton />);
    const skeletons = container.querySelectorAll("[aria-hidden='true']");
    // Should have many skeleton elements from nested components
    expect(skeletons.length).toBeGreaterThan(10);
  });
});
