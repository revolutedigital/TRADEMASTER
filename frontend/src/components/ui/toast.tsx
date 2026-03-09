"use client";

import { useState, useCallback, useRef, createContext, useContext, useEffect, type ReactNode } from "react";
import { X, CheckCircle, AlertCircle, AlertTriangle, Info } from "lucide-react";
import { cn } from "@/lib/utils";

type ToastType = "success" | "error" | "warning" | "info";

interface Toast {
  id: number;
  type: ToastType;
  message: string;
  createdAt: number;
}

interface ToastContextValue {
  success: (msg: string) => void;
  error: (msg: string) => void;
  warning: (msg: string) => void;
  info: (msg: string) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) {
    // Return no-op functions if not wrapped in provider (prevents crash)
    return {
      success: () => {},
      error: () => {},
      warning: () => {},
      info: () => {},
    };
  }
  return ctx;
}

const icons: Record<ToastType, typeof CheckCircle> = {
  success: CheckCircle,
  error: AlertCircle,
  warning: AlertTriangle,
  info: Info,
};

const styles: Record<ToastType, string> = {
  success: "border-green-500/20 bg-green-500/10 text-green-300",
  error: "border-red-500/20 bg-red-500/10 text-red-300",
  warning: "border-yellow-500/20 bg-yellow-500/10 text-yellow-300",
  info: "border-blue-500/20 bg-blue-500/10 text-blue-300",
};

const TOAST_DURATION = 5000;
const MAX_TOASTS = 5;

function ToastItem({ toast, onRemove }: { toast: Toast; onRemove: (id: number) => void }) {
  const Icon = icons[toast.type];
  const [progress, setProgress] = useState(100);
  const [exiting, setExiting] = useState(false);

  useEffect(() => {
    const start = toast.createdAt;
    const frame = () => {
      const elapsed = Date.now() - start;
      const remaining = Math.max(0, 100 - (elapsed / TOAST_DURATION) * 100);
      setProgress(remaining);
      if (remaining > 0) requestAnimationFrame(frame);
    };
    const raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [toast.createdAt]);

  const handleRemove = () => {
    setExiting(true);
    setTimeout(() => onRemove(toast.id), 200);
  };

  return (
    <div
      className={cn(
        "flex items-start gap-3 rounded-xl border px-4 py-3 text-sm shadow-lg backdrop-blur-sm",
        "transition-all duration-200",
        exiting ? "opacity-0 translate-x-4" : "animate-slide-in",
        styles[toast.type]
      )}
      role="alert"
    >
      <Icon className="h-4 w-4 shrink-0 mt-0.5" />
      <span className="flex-1 pr-2">{toast.message}</span>
      <button
        onClick={handleRemove}
        className="shrink-0 rounded-md p-0.5 hover:bg-white/10 transition-colors"
        aria-label="Dismiss notification"
      >
        <X className="h-3.5 w-3.5" />
      </button>
      {/* Progress bar */}
      <div className="absolute bottom-0 left-0 right-0 h-0.5 overflow-hidden rounded-b-xl">
        <div
          className="h-full bg-current opacity-30 transition-none"
          style={{ width: `${progress}%` }}
        />
      </div>
    </div>
  );
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const idRef = useRef(0);

  const addToast = useCallback((type: ToastType, message: string) => {
    const id = ++idRef.current;
    setToasts((prev) => [...prev.slice(-(MAX_TOASTS - 1)), { id, type, message, createdAt: Date.now() }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, TOAST_DURATION);
  }, []);

  const remove = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const value: ToastContextValue = {
    success: (msg) => addToast("success", msg),
    error: (msg) => addToast("error", msg),
    warning: (msg) => addToast("warning", msg),
    info: (msg) => addToast("info", msg),
  };

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div
        className="fixed bottom-4 right-4 z-[100] flex flex-col gap-2 max-w-sm w-full pointer-events-none"
        aria-live="polite"
        aria-label="Notifications"
      >
        {toasts.map((toast) => (
          <div key={toast.id} className="pointer-events-auto relative">
            <ToastItem toast={toast} onRemove={remove} />
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}
