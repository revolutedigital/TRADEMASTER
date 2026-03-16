import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

// Mock authStore
const mockLogin = vi.fn();
const mockAuthState = {
  login: mockLogin,
  isLoading: false,
  error: null as string | null,
};

vi.mock("@/stores/authStore", () => ({
  useAuthStore: vi.fn(() => mockAuthState),
}));

import { LoginPage } from "@/components/login-page";

describe("LoginPage", () => {
  beforeEach(() => {
    mockLogin.mockReset();
    mockAuthState.isLoading = false;
    mockAuthState.error = null;
    mockAuthState.login = mockLogin;
  });

  it("renders login form with username and password inputs", () => {
    render(<LoginPage />);
    // Use getByLabelText with exact match to avoid password toggle button collision
    expect(screen.getByLabelText("Username")).toBeInTheDocument();
    expect(screen.getByLabelText("Password")).toBeInTheDocument();
  });

  it("renders Welcome back heading", () => {
    render(<LoginPage />);
    expect(screen.getByText("Welcome back")).toBeInTheDocument();
  });

  it("renders Sign In button", () => {
    render(<LoginPage />);
    expect(screen.getByRole("button", { name: /sign in/i })).toBeInTheDocument();
  });

  it("calls login with credentials on submit", async () => {
    mockLogin.mockResolvedValue(true);
    render(<LoginPage />);

    fireEvent.change(screen.getByLabelText("Username"), {
      target: { value: "admin" },
    });
    fireEvent.change(screen.getByLabelText("Password"), {
      target: { value: "secretpass" },
    });
    fireEvent.click(screen.getByRole("button", { name: /sign in/i }));

    await waitFor(() => {
      expect(mockLogin).toHaveBeenCalledWith("admin", "secretpass");
    });
  });

  it("shows loading state when isLoading is true", () => {
    mockAuthState.isLoading = true;
    render(<LoginPage />);
    expect(screen.getByText(/signing in/i)).toBeInTheDocument();
  });

  it("shows error message when error exists", () => {
    mockAuthState.error = "Invalid credentials";
    render(<LoginPage />);
    expect(screen.getByText("Invalid credentials")).toBeInTheDocument();
  });

  it("shows Paper Trading Mode text in footer", () => {
    render(<LoginPage />);
    // Use getAllByText and check at least one exists
    const elements = screen.getAllByText(/paper trading mode/i);
    expect(elements.length).toBeGreaterThan(0);
  });

  it("requires both fields before submit", () => {
    render(<LoginPage />);
    const username = screen.getByLabelText("Username");
    const password = screen.getByLabelText("Password");
    expect(username).toBeRequired();
    expect(password).toBeRequired();
  });
});
