import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";

const mockAuthState = {
  isAuthenticated: false,
  restoreSession: vi.fn(),
};

vi.mock("@/stores/authStore", () => ({
  useAuthStore: vi.fn(() => mockAuthState),
}));

// Mock LoginPage
vi.mock("@/components/login-page", () => ({
  LoginPage: () => <div data-testid="login-page">Login Page</div>,
}));

import { AuthGuard } from "@/components/auth-guard";

describe("AuthGuard", () => {
  beforeEach(() => {
    mockAuthState.isAuthenticated = false;
    mockAuthState.restoreSession.mockReset();
  });

  it("calls restoreSession on mount", () => {
    render(<AuthGuard><div>Protected</div></AuthGuard>);
    expect(mockAuthState.restoreSession).toHaveBeenCalled();
  });

  it("shows login page when not authenticated", () => {
    render(<AuthGuard><div>Protected</div></AuthGuard>);
    expect(screen.getByTestId("login-page")).toBeInTheDocument();
    expect(screen.queryByText("Protected")).toBeNull();
  });

  it("shows children when authenticated", () => {
    mockAuthState.isAuthenticated = true;
    render(<AuthGuard><div>Protected Content</div></AuthGuard>);
    expect(screen.getByText("Protected Content")).toBeInTheDocument();
  });
});
