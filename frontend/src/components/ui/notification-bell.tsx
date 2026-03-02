"use client";

import { useEffect, useRef } from "react";
import { useNotificationStore } from "@/stores/notificationStore";
import { cn } from "@/lib/utils";
import { Bell, Check, CheckCheck } from "lucide-react";

export function NotificationBell() {
  const { notifications, unreadCount, isOpen, setOpen, fetchCount, markAsRead, markAllRead } =
    useNotificationStore();
  const ref = useRef<HTMLDivElement>(null);

  // Poll unread count every 30 seconds
  useEffect(() => {
    fetchCount();
    const interval = setInterval(fetchCount, 30_000);
    return () => clearInterval(interval);
  }, [fetchCount]);

  // Close dropdown on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    if (isOpen) document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [isOpen, setOpen]);

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(!isOpen)}
        className="relative p-1.5 rounded-md text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface-hover)] transition-colors"
        aria-label={`Notifications${unreadCount > 0 ? `, ${unreadCount} unread` : ""}`}
      >
        <Bell className="h-4 w-4" />
        {unreadCount > 0 && (
          <span className="absolute -top-0.5 -right-0.5 h-4 w-4 rounded-full bg-[var(--color-danger)] text-[10px] font-bold text-white flex items-center justify-center">
            {unreadCount > 9 ? "9+" : unreadCount}
          </span>
        )}
      </button>

      {isOpen && (
        <div className="absolute right-0 top-full mt-2 w-80 max-h-96 overflow-y-auto rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] shadow-xl z-50">
          {/* Header */}
          <div className="flex items-center justify-between border-b border-[var(--color-border)] px-4 py-3">
            <span className="text-sm font-semibold">Notifications</span>
            {unreadCount > 0 && (
              <button
                onClick={markAllRead}
                className="flex items-center gap-1 text-xs text-[var(--color-primary)] hover:underline"
              >
                <CheckCheck className="h-3 w-3" />
                Mark all read
              </button>
            )}
          </div>

          {/* List */}
          {notifications.length === 0 ? (
            <p className="px-4 py-8 text-center text-sm text-[var(--color-text-muted)]">
              No notifications yet
            </p>
          ) : (
            <div className="divide-y divide-[var(--color-border)]">
              {notifications.map((n) => (
                <div
                  key={n.id}
                  className={cn(
                    "px-4 py-3 text-sm cursor-pointer hover:bg-[var(--color-surface-hover)] transition-colors",
                    !n.is_read && "bg-[var(--color-primary)]/5"
                  )}
                  onClick={() => !n.is_read && markAsRead(n.id)}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span
                          className={cn(
                            "inline-block h-2 w-2 rounded-full flex-shrink-0",
                            n.severity === "error" && "bg-red-500",
                            n.severity === "warning" && "bg-yellow-500",
                            n.severity === "success" && "bg-green-500",
                            n.severity === "info" && "bg-blue-500"
                          )}
                        />
                        <span className="font-medium truncate">{n.title}</span>
                      </div>
                      <p className="mt-0.5 text-xs text-[var(--color-text-muted)] line-clamp-2">
                        {n.message}
                      </p>
                    </div>
                    {!n.is_read && (
                      <Check className="h-3.5 w-3.5 text-[var(--color-text-muted)] flex-shrink-0 mt-0.5" />
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
