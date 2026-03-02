"use client";

import { useState, useEffect, useRef } from "react";

interface CursorPosition {
  userId: string;
  username: string;
  x: number;
  y: number;
  color: string;
  lastUpdate: number;
}

const CURSOR_COLORS = ["#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6", "#ec4899"];

export function LiveCursors() {
  const [cursors, setCursors] = useState<Map<string, CursorPosition>>(new Map());
  const wsRef = useRef<WebSocket | null>(null);
  const cursorTimeout = 10000; // Remove cursors after 10s inactivity

  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/cursors`);
    wsRef.current = ws;

    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type === "cursor_move") {
          setCursors((prev) => {
            const next = new Map(prev);
            next.set(data.userId, {
              userId: data.userId,
              username: data.username,
              x: data.x,
              y: data.y,
              color: CURSOR_COLORS[data.userId.charCodeAt(0) % CURSOR_COLORS.length],
              lastUpdate: Date.now(),
            });
            return next;
          });
        }
      } catch {}
    };

    // Send own cursor position
    const handleMouseMove = (e: MouseEvent) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          type: "cursor_move",
          x: e.clientX / window.innerWidth,
          y: e.clientY / window.innerHeight,
        }));
      }
    };

    window.addEventListener("mousemove", handleMouseMove);

    // Cleanup stale cursors
    const cleanup = setInterval(() => {
      setCursors((prev) => {
        const next = new Map(prev);
        const now = Date.now();
        for (const [id, cursor] of next) {
          if (now - cursor.lastUpdate > cursorTimeout) next.delete(id);
        }
        return next;
      });
    }, 5000);

    return () => {
      ws.close();
      window.removeEventListener("mousemove", handleMouseMove);
      clearInterval(cleanup);
    };
  }, []);

  return (
    <div className="fixed inset-0 pointer-events-none z-50">
      {Array.from(cursors.values()).map((cursor) => (
        <div
          key={cursor.userId}
          className="absolute transition-all duration-100 ease-out"
          style={{
            left: `${cursor.x * 100}%`,
            top: `${cursor.y * 100}%`,
            transform: "translate(-2px, -2px)",
          }}
        >
          {/* Cursor arrow */}
          <svg width="16" height="20" viewBox="0 0 16 20" fill="none">
            <path d="M0 0L16 12H6L0 20V0Z" fill={cursor.color} stroke="white" strokeWidth="1" />
          </svg>
          {/* Username label */}
          <div
            className="absolute left-4 top-4 px-2 py-0.5 rounded text-xs text-white whitespace-nowrap"
            style={{ backgroundColor: cursor.color }}
          >
            {cursor.username}
          </div>
        </div>
      ))}
    </div>
  );
}
