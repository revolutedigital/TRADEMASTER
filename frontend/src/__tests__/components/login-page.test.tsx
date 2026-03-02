import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

// Mock authStore
const mockLogin = vi.fn();
vi.mock("@/stores/authStore", () => ({
  useAuthStore: vi.fn(() => ({
    login: mockLogin,
    isLoading: false,
    error: null,
  })),
}));

import { LoginPage } from "@/components/login-page";

describe("LoginPage", () => {
  beforeEach(() => {
    mockLogin.mockReset();
  });

  it("renders login form with username and password fields", () => {
    render(<LoginPage />);
    expect(screen.getByLabelText(/username/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
  });

  it("renders TradeMaster heading", () => {
    render(<LoginPage />);
    expect(screen.getByText("TradeMaster")).toBeInTheDocument();
  });

  it("renders Sign In button", () => {
    render(<LoginPage />);
    expect(screen.getByRole("button", { name: /sign in/i })).toBeInTheDocument();
  });

  it("calls login with credentials on submit", async () => {
    mockLogin.mockResolvedValue(true);
    render(<LoginPage />);

    fireEvent.change(screen.getByLabelText(/username/i), {
      target: { value: "admin" },
    });
    fireEvent.change(screen.getByLabelText(/password/i), {
      target: { value: "secretpass" },
    });
    fireEvent.click(screen.getByRole("button", { name: /sign in/i }));

    await waitFor(() => {
      expect(mockLogin).toHaveBeenCalledWith("admin", "secretpass");
    });
  });

  it("shows loading state when isLoading is true", () => {
    const { useAuthStore } = require("@/stores/authStore");
    (useAuthStore as ReturnType<typeof vi.fn>).mockReturnValue({
      login: mockLogin,
      isLoading: true,
      error: null,
    });

    render(<LoginPage />);
    expect(screen.getByText(/signing in/i)).toBeInTheDocument();
  });

  it("shows error message when error exists", () => {
    const { useAuthStore } = require("@/stores/authStore");
    (useAuthStore as ReturnType<typeof vi.fn>).mockReturnValue({
      login: mockLogin,
      isLoading: false,
      error: "Invalid credentials",
    });

    render(<LoginPage />);
    expect(screen.getByText("Invalid credentials")).toBeInTheDocument();
  });

  it("shows Paper Trading Mode Active text", () => {
    render(<LoginPage />);
    expect(screen.getByText(/paper trading mode active/i)).toBeInTheDocument();
  });

  it("requires both fields before submit", () => {
    render(<LoginPage />);
    const username = screen.getByLabelText(/username/i);
    const password = screen.getByLabelText(/password/i);
    expect(username).toBeRequired();
    expect(password).toBeRequired();
  });
});
