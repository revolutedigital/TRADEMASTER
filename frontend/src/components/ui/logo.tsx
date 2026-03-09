import { cn } from "@/lib/utils";

interface LogoProps {
  variant?: "full" | "icon";
  size?: "sm" | "md" | "lg";
  className?: string;
}

const sizes = {
  sm: { icon: "h-6 w-6", text: "text-base", gap: "gap-1.5" },
  md: { icon: "h-8 w-8", text: "text-lg", gap: "gap-2" },
  lg: { icon: "h-10 w-10", text: "text-2xl", gap: "gap-2.5" },
};

function LogoIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 40 40" fill="none" className={className} aria-hidden="true">
      <defs>
        <linearGradient id="lg" x1="0" y1="0" x2="40" y2="40" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#6366f1" />
          <stop offset="100%" stopColor="#8b5cf6" />
        </linearGradient>
      </defs>
      <rect x="4" y="16" width="5" height="14" rx="2.5" fill="url(#lg)" />
      <rect x="12" y="8" width="5" height="22" rx="2.5" fill="url(#lg)" />
      <rect x="20" y="12" width="5" height="16" rx="2.5" fill="url(#lg)" />
      <rect x="28" y="6" width="5" height="20" rx="2.5" fill="url(#lg)" />
      <circle cx="35" cy="10" r="3" fill="#818cf8" />
      <circle cx="35" cy="10" r="5" fill="none" stroke="#818cf8" strokeWidth="1" opacity="0.3" />
    </svg>
  );
}

export function Logo({ variant = "full", size = "md", className }: LogoProps) {
  const s = sizes[size];

  if (variant === "icon") {
    return <LogoIcon className={cn(s.icon, className)} />;
  }

  return (
    <div className={cn("flex items-center", s.gap, className)} aria-label="TradeMaster">
      <LogoIcon className={s.icon} />
      <span className={cn(s.text, "font-bold tracking-tight")}>
        Trade<span className="text-gradient">Master</span>
      </span>
    </div>
  );
}
