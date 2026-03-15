"use client";

import React from "react";

interface Shortcut {
  key: string;
  ctrl?: boolean;
  shift?: boolean;
  description: string;
}

interface ShortcutHelpProps {
  isOpen: boolean;
  onClose: () => void;
  shortcuts: Shortcut[];
}

function formatKey(shortcut: Shortcut): string {
  const parts: string[] = [];
  if (shortcut.ctrl) parts.push("Ctrl");
  if (shortcut.shift) parts.push("Shift");
  parts.push(shortcut.key.toUpperCase());
  return parts.join(" + ");
}

export function ShortcutHelp({ isOpen, onClose, shortcuts }: ShortcutHelpProps) {
  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
      onClick={onClose}
    >
      <div
        className="w-full max-w-md rounded-lg border border-slate-700 bg-slate-800 p-6 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-white">Keyboard Shortcuts</h2>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white"
          >
            ESC
          </button>
        </div>
        <div className="space-y-2">
          {shortcuts.map((shortcut, i) => (
            <div
              key={i}
              className="flex items-center justify-between rounded px-3 py-2 hover:bg-slate-700/50"
            >
              <span className="text-sm text-slate-300">
                {shortcut.description}
              </span>
              <kbd className="rounded border border-slate-600 bg-slate-700 px-2 py-1 text-xs font-mono text-slate-300">
                {formatKey(shortcut)}
              </kbd>
            </div>
          ))}
          <div className="flex items-center justify-between rounded px-3 py-2 hover:bg-slate-700/50">
            <span className="text-sm text-slate-300">Show this help</span>
            <kbd className="rounded border border-slate-600 bg-slate-700 px-2 py-1 text-xs font-mono text-slate-300">
              ?
            </kbd>
          </div>
        </div>
      </div>
    </div>
  );
}
