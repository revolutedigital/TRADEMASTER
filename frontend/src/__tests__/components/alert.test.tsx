import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { Alert } from "@/components/ui/alert";

describe("Alert", () => {
  it("renders children content", () => {
    render(<Alert>This is an alert message</Alert>);
    expect(screen.getByText("This is an alert message")).toBeInTheDocument();
  });

  it("renders title when provided", () => {
    render(<Alert title="Warning">Content</Alert>);
    expect(screen.getByText("Warning")).toBeInTheDocument();
  });

  it("renders with info variant by default (status role)", () => {
    render(<Alert>Info message</Alert>);
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("renders with alert role for error variant", () => {
    render(<Alert variant="error">Error occurred</Alert>);
    expect(screen.getByRole("alert")).toBeInTheDocument();
  });

  it("can be dismissed when dismissible", () => {
    render(<Alert dismissible>Dismissible alert</Alert>);
    const dismissBtn = screen.getByLabelText("Dismiss");
    fireEvent.click(dismissBtn);
    expect(screen.queryByText("Dismissible alert")).toBeNull();
  });

  it("does not show dismiss button when not dismissible", () => {
    render(<Alert>Non-dismissible</Alert>);
    expect(screen.queryByLabelText("Dismiss")).toBeNull();
  });
});
