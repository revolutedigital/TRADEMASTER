"use client";

export default function SettingsError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-4 p-8">
      <div className="rounded-lg border border-red-500/20 bg-red-500/5 p-6 max-w-md text-center">
        <h2 className="text-xl font-bold text-red-400 mb-2">Settings Error</h2>
        <p className="text-sm text-muted-foreground mb-4">{error.message}</p>
        <button
          onClick={reset}
          className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors"
        >
          Try Again
        </button>
      </div>
    </div>
  );
}
