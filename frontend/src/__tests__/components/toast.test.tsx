import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import { ToastProvider, useToast } from "@/components/ui/toast";

// Component to trigger toasts inside the provider
function ToastTrigger() {
  const toast = useToast();
  return (
    <div>
      <button onClick={() => toast.success("Success message")}>Show Success</button>
      <button onClick={() => toast.error("Error message")}>Show Error</button>
      <button onClick={() => toast.warning("Warning message")}>Show Warning</button>
      <button onClick={() => toast.info("Info message")}>Show Info</button>
    </div>
  );
}

describe("ToastProvider", () => {
  it("renders children", () => {
    render(
      <ToastProvider>
        <div>Child content</div>
      </ToastProvider>
    );
    expect(screen.getByText("Child content")).toBeInTheDocument();
  });

  it("shows success toast when triggered", () => {
    render(
      <ToastProvider>
        <ToastTrigger />
      </ToastProvider>
    );
    fireEvent.click(screen.getByText("Show Success"));
    expect(screen.getByText("Success message")).toBeInTheDocument();
  });

  it("shows error toast when triggered", () => {
    render(
      <ToastProvider>
        <ToastTrigger />
      </ToastProvider>
    );
    fireEvent.click(screen.getByText("Show Error"));
    expect(screen.getByText("Error message")).toBeInTheDocument();
  });

  it("toasts have alert role", () => {
    render(
      <ToastProvider>
        <ToastTrigger />
      </ToastProvider>
    );
    fireEvent.click(screen.getByText("Show Warning"));
    const alerts = screen.getAllByRole("alert");
    expect(alerts.length).toBeGreaterThan(0);
  });

  it("toast has dismiss button", () => {
    render(
      <ToastProvider>
        <ToastTrigger />
      </ToastProvider>
    );
    fireEvent.click(screen.getByText("Show Info"));
    expect(screen.getByLabelText("Dismiss notification")).toBeInTheDocument();
  });
});

describe("useToast outside provider", () => {
  it("returns no-op functions when not wrapped in provider", () => {
    function Naked() {
      const toast = useToast();
      // Should not throw
      toast.success("test");
      toast.error("test");
      toast.warning("test");
      toast.info("test");
      return <div>OK</div>;
    }
    render(<Naked />);
    expect(screen.getByText("OK")).toBeInTheDocument();
  });
});
