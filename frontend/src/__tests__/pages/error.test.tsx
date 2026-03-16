import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import GlobalError from "@/app/error";

describe("GlobalError", () => {
  it("renders error message", () => {
    const error = new Error("Something broke");
    render(<GlobalError error={error} reset={vi.fn()} />);
    expect(screen.getByText("Something went wrong")).toBeInTheDocument();
    expect(screen.getByText("Something broke")).toBeInTheDocument();
  });

  it("renders Try Again button", () => {
    const error = new Error("Oops");
    render(<GlobalError error={error} reset={vi.fn()} />);
    expect(screen.getByText("Try Again")).toBeInTheDocument();
  });

  it("calls reset when Try Again is clicked", () => {
    const reset = vi.fn();
    const error = new Error("Oops");
    render(<GlobalError error={error} reset={reset} />);
    fireEvent.click(screen.getByText("Try Again"));
    expect(reset).toHaveBeenCalledOnce();
  });
});
