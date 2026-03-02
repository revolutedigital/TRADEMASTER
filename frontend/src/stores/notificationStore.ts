import { create } from "zustand";
import { apiFetch } from "@/lib/utils";

interface Notification {
  id: number;
  type: string;
  title: string;
  message: string;
  severity: string;
  is_read: boolean;
  created_at: string;
}

interface NotificationState {
  notifications: Notification[];
  unreadCount: number;
  isOpen: boolean;
  setOpen: (open: boolean) => void;
  fetchNotifications: () => Promise<void>;
  fetchCount: () => Promise<void>;
  markAsRead: (id: number) => Promise<void>;
  markAllRead: () => Promise<void>;
}

export const useNotificationStore = create<NotificationState>((set, get) => ({
  notifications: [],
  unreadCount: 0,
  isOpen: false,

  setOpen: (open: boolean) => {
    set({ isOpen: open });
    if (open) get().fetchNotifications();
  },

  fetchNotifications: async () => {
    try {
      const data = await apiFetch<Notification[]>("/api/v1/notifications/?limit=20");
      set({ notifications: data });
    } catch {
      // silent fail
    }
  },

  fetchCount: async () => {
    try {
      const data = await apiFetch<{ unread: number }>("/api/v1/notifications/count");
      set({ unreadCount: data.unread });
    } catch {
      // silent fail
    }
  },

  markAsRead: async (id: number) => {
    try {
      await apiFetch(`/api/v1/notifications/${id}/read`, { method: "POST" });
      set((state) => ({
        notifications: state.notifications.map((n) =>
          n.id === id ? { ...n, is_read: true } : n
        ),
        unreadCount: Math.max(0, state.unreadCount - 1),
      }));
    } catch {
      // silent fail
    }
  },

  markAllRead: async () => {
    try {
      await apiFetch("/api/v1/notifications/read-all", { method: "POST" });
      set((state) => ({
        notifications: state.notifications.map((n) => ({ ...n, is_read: true })),
        unreadCount: 0,
      }));
    } catch {
      // silent fail
    }
  },
}));
