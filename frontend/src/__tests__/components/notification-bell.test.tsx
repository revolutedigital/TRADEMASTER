import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

const mockNotifState = {
  notifications: [] as { id: string; title: string; message: string; severity: string; is_read: boolean }[],
  unreadCount: 0,
  isOpen: false,
  setOpen: vi.fn(),
  fetchCount: vi.fn(),
  markAsRead: vi.fn(),
  markAllRead: vi.fn(),
};

vi.mock("@/stores/notificationStore", () => ({
  useNotificationStore: () => mockNotifState,
}));

import { NotificationBell } from "@/components/ui/notification-bell";

describe("NotificationBell", () => {
  beforeEach(() => {
    mockNotifState.notifications = [];
    mockNotifState.unreadCount = 0;
    mockNotifState.isOpen = false;
    vi.clearAllMocks();
  });

  it("renders bell button", () => {
    render(<NotificationBell />);
    expect(screen.getByLabelText("Notifications")).toBeInTheDocument();
  });

  it("shows unread count badge when > 0", () => {
    mockNotifState.unreadCount = 3;
    render(<NotificationBell />);
    expect(screen.getByText("3")).toBeInTheDocument();
    expect(screen.getByLabelText("Notifications, 3 unread")).toBeInTheDocument();
  });

  it("shows 9+ for counts above 9", () => {
    mockNotifState.unreadCount = 15;
    render(<NotificationBell />);
    expect(screen.getByText("9+")).toBeInTheDocument();
  });

  it("calls setOpen on bell click", () => {
    render(<NotificationBell />);
    fireEvent.click(screen.getByLabelText("Notifications"));
    expect(mockNotifState.setOpen).toHaveBeenCalledWith(true);
  });

  it("shows dropdown with Notificações when isOpen", () => {
    mockNotifState.isOpen = true;
    render(<NotificationBell />);
    expect(screen.getByText("Notificações")).toBeInTheDocument();
  });

  it("shows empty state when no notifications", () => {
    mockNotifState.isOpen = true;
    render(<NotificationBell />);
    expect(screen.getByText("Nenhuma notificação ainda")).toBeInTheDocument();
  });

  it("renders notification items when present", () => {
    mockNotifState.isOpen = true;
    mockNotifState.unreadCount = 1;
    mockNotifState.notifications = [
      { id: "1", title: "Alert", message: "BTC dropped", severity: "warning", is_read: false },
    ];
    render(<NotificationBell />);
    expect(screen.getByText("Alert")).toBeInTheDocument();
    expect(screen.getByText("BTC dropped")).toBeInTheDocument();
  });

  it("shows mark all read button when unread > 0", () => {
    mockNotifState.isOpen = true;
    mockNotifState.unreadCount = 2;
    mockNotifState.notifications = [
      { id: "1", title: "A", message: "m", severity: "info", is_read: false },
    ];
    render(<NotificationBell />);
    expect(screen.getByText("Marcar todas como lidas")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Marcar todas como lidas"));
    expect(mockNotifState.markAllRead).toHaveBeenCalled();
  });

  it("calls fetchCount on mount", () => {
    render(<NotificationBell />);
    expect(mockNotifState.fetchCount).toHaveBeenCalled();
  });
});
