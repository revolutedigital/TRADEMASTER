import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock apiFetch at the module level using vi.hoisted
const mockApiFetch = vi.hoisted(() => vi.fn());

vi.mock("@/lib/utils", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/utils")>();
  return {
    ...actual,
    apiFetch: mockApiFetch,
  };
});

import { useNotificationStore } from "@/stores/notificationStore";

describe("notificationStore", () => {
  beforeEach(() => {
    mockApiFetch.mockReset();
    // Reset store state
    useNotificationStore.setState({
      notifications: [],
      unreadCount: 0,
      isOpen: false,
    });
  });

  it("initializes with empty notifications", () => {
    const state = useNotificationStore.getState();
    expect(state.notifications).toEqual([]);
    expect(state.unreadCount).toBe(0);
    expect(state.isOpen).toBe(false);
  });

  it("setOpen toggles isOpen and fetches when opening", async () => {
    mockApiFetch.mockResolvedValueOnce([]);
    useNotificationStore.getState().setOpen(true);
    expect(useNotificationStore.getState().isOpen).toBe(true);
    expect(mockApiFetch).toHaveBeenCalledWith(expect.stringContaining("/notifications/"));
  });

  it("fetchCount updates unread count", async () => {
    mockApiFetch.mockResolvedValueOnce({ unread: 5 });
    await useNotificationStore.getState().fetchCount();
    expect(useNotificationStore.getState().unreadCount).toBe(5);
  });

  it("markAsRead updates notification and decrements count", async () => {
    useNotificationStore.setState({
      notifications: [
        { id: 1, type: "trade", title: "Trade", message: "Executed", severity: "info", is_read: false, created_at: "2024-01-01" },
      ],
      unreadCount: 1,
    });

    mockApiFetch.mockResolvedValueOnce({});
    await useNotificationStore.getState().markAsRead(1);

    const state = useNotificationStore.getState();
    expect(state.notifications[0].is_read).toBe(true);
    expect(state.unreadCount).toBe(0);
  });

  it("markAllRead marks all notifications as read", async () => {
    useNotificationStore.setState({
      notifications: [
        { id: 1, type: "trade", title: "T1", message: "M1", severity: "info", is_read: false, created_at: "2024-01-01" },
        { id: 2, type: "alert", title: "T2", message: "M2", severity: "warning", is_read: false, created_at: "2024-01-02" },
      ],
      unreadCount: 2,
    });

    mockApiFetch.mockResolvedValueOnce({});
    await useNotificationStore.getState().markAllRead();

    const state = useNotificationStore.getState();
    expect(state.notifications.every((n) => n.is_read)).toBe(true);
    expect(state.unreadCount).toBe(0);
  });
});
