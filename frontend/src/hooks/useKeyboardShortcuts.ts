"use client";

import { useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";

interface ShortcutConfig {
  key: string;
  ctrl?: boolean;
  shift?: boolean;
  description: string;
  action: () => void;
}

export function useKeyboardShortcuts() {
  const router = useRouter();
  const [showHelp, setShowHelp] = useState(false);

  const shortcuts: ShortcutConfig[] = [
    { key: "d", ctrl: true, description: "Go to Dashboard", action: () => router.push("/") },
    { key: "t", ctrl: true, description: "Go to Trading", action: () => router.push("/trading") },
    { key: "p", ctrl: true, description: "Go to Portfolio", action: () => router.push("/portfolio") },
    { key: "b", ctrl: true, description: "Go to Backtest", action: () => router.push("/backtest") },
    { key: "s", ctrl: true, shift: true, description: "Go to Settings", action: () => router.push("/settings") },
    { key: "?", description: "Show keyboard shortcuts", action: () => setShowHelp((v) => !v) },
    { key: "Escape", description: "Close modals", action: () => setShowHelp(false) },
  ];

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      // Don't trigger shortcuts when typing in inputs
      const target = e.target as HTMLElement;
      if (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable) {
        return;
      }

      for (const shortcut of shortcuts) {
        const ctrlMatch = shortcut.ctrl ? (e.ctrlKey || e.metaKey) : true;
        const shiftMatch = shortcut.shift ? e.shiftKey : !e.shiftKey;
        if (e.key === shortcut.key && ctrlMatch && shiftMatch) {
          e.preventDefault();
          shortcut.action();
          return;
        }
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [router]
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  return { showHelp, setShowHelp, shortcuts };
}
