import { TableSkeleton } from "@/components/ui/skeleton";

export default function SignalsLoading() {
  return (
    <div className="space-y-6">
      <div className="h-8 w-36 animate-pulse rounded bg-muted/50" />
      <TableSkeleton rows={8} />
    </div>
  );
}
