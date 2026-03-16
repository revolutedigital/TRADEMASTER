import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { Dialog, DialogFooter } from "@/components/ui/dialog";

describe("Dialog", () => {
  it("returns null when not open", () => {
    const { container } = render(
      <Dialog open={false} onClose={() => {}}>Content</Dialog>
    );
    expect(container.innerHTML).toBe("");
  });

  it("renders content in a dialog role when open", () => {
    render(
      <Dialog open={true} onClose={() => {}}>
        <p>Dialog content</p>
      </Dialog>
    );
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByText("Dialog content")).toBeInTheDocument();
  });

  it("renders title and description when provided", () => {
    render(
      <Dialog open={true} onClose={() => {}} title="My Title" description="My Desc">
        Body
      </Dialog>
    );
    expect(screen.getByText("My Title")).toBeInTheDocument();
    expect(screen.getByText("My Desc")).toBeInTheDocument();
  });

  it("has a close button with aria-label", () => {
    render(
      <Dialog open={true} onClose={() => {}}>Content</Dialog>
    );
    expect(screen.getByLabelText("Close dialog")).toBeInTheDocument();
  });

  it("calls onClose when close button is clicked", () => {
    const onClose = vi.fn();
    render(
      <Dialog open={true} onClose={onClose}>Content</Dialog>
    );
    fireEvent.click(screen.getByLabelText("Close dialog"));
    expect(onClose).toHaveBeenCalled();
  });

  it("calls onClose on Escape key", () => {
    const onClose = vi.fn();
    render(
      <Dialog open={true} onClose={onClose}>Content</Dialog>
    );
    fireEvent.keyDown(document, { key: "Escape" });
    expect(onClose).toHaveBeenCalled();
  });
});

describe("DialogFooter", () => {
  it("renders children", () => {
    render(<DialogFooter><button>OK</button></DialogFooter>);
    expect(screen.getByText("OK")).toBeInTheDocument();
  });
});
