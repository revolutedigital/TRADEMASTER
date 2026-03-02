"use client";

import { X } from "lucide-react";

interface ShortcutHelpProps {
  shortcuts: Array<{ key: string; ctrl?: boolean; shift?: boolean; description: string }>;
  onClose: () => void;
}

export function ShortcutHelp({ shortcuts, onClose }: ShortcutHelpProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={onClose}>
      <div
        className="w-full max-w-sm rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-lg font-semibold">Keyboard Shortcuts</h3>
          <button onClick={onClose} className="rounded p-1 hover:bg-[var(--color-surface-hover)]">
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="space-y-2">
          {shortcuts.map((s, i) => (
            <div key={i} className="flex items-center justify-between text-sm">
              <span className="text-[var(--color-text-muted)]">{s.description}</span>
              <kbd className="rounded bg-[var(--color-background)] px-2 py-0.5 font-mono text-xs">
                {s.ctrl ? "Ctrl+" : ""}
                {s.shift ? "Shift+" : ""}
                {s.key}
              </kbd>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
