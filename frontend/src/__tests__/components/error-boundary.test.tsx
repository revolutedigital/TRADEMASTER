import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ErrorBoundary } from "@/components/ui/error-boundary";

// Component that throws
function ThrowError({ shouldThrow }: { shouldThrow: boolean }) {
  if (shouldThrow) throw new Error("Test error message");
  return <div>Normal content</div>;
}

describe("ErrorBoundary", () => {
  // Suppress console.error for expected errors
  const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});

  beforeEach(() => {
    consoleSpy.mockClear();
  });

  it("renders children when no error", () => {
    render(
      <ErrorBoundary>
        <ThrowError shouldThrow={false} />
      </ErrorBoundary>
    );
    expect(screen.getByText("Normal content")).toBeInTheDocument();
  });

  it("renders error UI when child throws", () => {
    render(
      <ErrorBoundary>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );
    expect(screen.getByText("Algo deu errado")).toBeInTheDocument();
  });

  it("renders custom fallback when provided", () => {
    render(
      <ErrorBoundary fallback={<div>Custom error page</div>}>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );
    expect(screen.getByText("Custom error page")).toBeInTheDocument();
  });

  it("shows error details on button click", () => {
    render(
      <ErrorBoundary>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );
    fireEvent.click(screen.getByText("Ver Detalhes"));
    expect(screen.getByText(/Test error message/)).toBeInTheDocument();
  });

  it("toggles error details visibility", () => {
    render(
      <ErrorBoundary>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );
    // Show details
    fireEvent.click(screen.getByText("Ver Detalhes"));
    expect(screen.getByText("Ocultar Detalhes")).toBeInTheDocument();
    // Hide details
    fireEvent.click(screen.getByText("Ocultar Detalhes"));
    expect(screen.getByText("Ver Detalhes")).toBeInTheDocument();
  });
});
