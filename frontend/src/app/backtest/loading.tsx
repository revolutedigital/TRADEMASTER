import { CardSkeleton, ChartSkeleton } from "@/components/ui/skeleton";

export default function BacktestLoading() {
  return (
    <div className="space-y-6">
      <div className="h-8 w-40 animate-pulse rounded bg-muted/50" />
      <CardSkeleton />
      <ChartSkeleton />
    </div>
  );
}
