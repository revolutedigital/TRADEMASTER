"use client";

import { useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";

interface ShortcutConfig {
  key: string;
  ctrl?: boolean;
  shift?: boolean;
  alt?: boolean;
  action: () => void;
  description: string;
}

export function useKeyboardShortcuts() {
  const router = useRouter();

  const shortcuts: ShortcutConfig[] = [
    {
      key: "d",
      ctrl: true,
      action: () => router.push("/"),
      description: "Go to Dashboard",
    },
    {
      key: "t",
      ctrl: true,
      action: () => router.push("/trading"),
      description: "Go to Trading",
    },
    {
      key: "p",
      ctrl: true,
      shift: true,
      action: () => router.push("/portfolio"),
      description: "Go to Portfolio",
    },
    {
      key: "b",
      ctrl: true,
      shift: true,
      action: () => router.push("/backtest"),
      description: "Go to Backtest",
    },
    {
      key: "s",
      ctrl: true,
      shift: true,
      action: () => router.push("/settings"),
      description: "Go to Settings",
    },
  ];

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      // Don't trigger in input fields
      const target = e.target as HTMLElement;
      if (
        target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.tagName === "SELECT" ||
        target.isContentEditable
      ) {
        return;
      }

      for (const shortcut of shortcuts) {
        const ctrlMatch = shortcut.ctrl ? e.ctrlKey || e.metaKey : !e.ctrlKey && !e.metaKey;
        const shiftMatch = shortcut.shift ? e.shiftKey : !e.shiftKey;
        const altMatch = shortcut.alt ? e.altKey : !e.altKey;

        if (
          e.key.toLowerCase() === shortcut.key.toLowerCase() &&
          ctrlMatch &&
          shiftMatch &&
          altMatch
        ) {
          e.preventDefault();
          shortcut.action();
          return;
        }
      }
    },
    [shortcuts]
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  return shortcuts;
}
