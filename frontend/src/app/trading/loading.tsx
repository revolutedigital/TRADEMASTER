import { ChartSkeleton, TableSkeleton } from "@/components/ui/skeleton";

export default function TradingLoading() {
  return (
    <div className="space-y-6">
      <div className="h-8 w-48 animate-pulse rounded bg-muted/50" />
      <ChartSkeleton />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <TableSkeleton rows={4} />
        <TableSkeleton rows={4} />
      </div>
    </div>
  );
}
