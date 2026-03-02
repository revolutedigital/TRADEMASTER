import { CardSkeleton, TableSkeleton } from "@/components/ui/skeleton";

export default function PortfolioLoading() {
  return (
    <div className="space-y-6">
      <div className="h-8 w-36 animate-pulse rounded bg-muted/50" />
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
        {Array.from({ length: 5 }).map((_, i) => (
          <CardSkeleton key={i} />
        ))}
      </div>
      <TableSkeleton rows={5} />
    </div>
  );
}
