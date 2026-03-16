import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { StatCard } from "@/components/ui/stat-card";
import { Progress, Spinner } from "@/components/ui/progress";
import { EmptyState } from "@/components/ui/empty-state";
import { Input } from "@/components/ui/input";
import { Skeleton, CardSkeleton, TableSkeleton } from "@/components/ui/skeleton";

describe("Card", () => {
  it("renders children", () => {
    render(<Card>Test Content</Card>);
    expect(screen.getByText("Test Content")).toBeInTheDocument();
  });

  it("applies default variant styles", () => {
    const { container } = render(<Card>Content</Card>);
    const card = container.firstChild as HTMLElement;
    expect(card.className).toContain("rounded-xl");
  });

  it("renders with glass variant", () => {
    const { container } = render(<Card variant="glass">Content</Card>);
    const card = container.firstChild as HTMLElement;
    expect(card.className).toContain("glass");
  });
});

describe("CardHeader & CardTitle", () => {
  it("renders header with title", () => {
    render(
      <CardHeader>
        <CardTitle>My Title</CardTitle>
      </CardHeader>
    );
    expect(screen.getByText("My Title")).toBeInTheDocument();
  });
});

describe("Badge", () => {
  it("renders with default variant", () => {
    render(<Badge>Active</Badge>);
    expect(screen.getByText("Active")).toBeInTheDocument();
  });

  it("renders with success variant", () => {
    const { container } = render(<Badge variant="success">Win</Badge>);
    const badge = container.firstChild as HTMLElement;
    expect(badge.className).toContain("green");
  });

  it("renders with danger variant", () => {
    const { container } = render(<Badge variant="danger">Loss</Badge>);
    const badge = container.firstChild as HTMLElement;
    expect(badge.className).toContain("red");
  });
});

describe("Button", () => {
  it("renders with children text", () => {
    render(<Button>Click me</Button>);
    expect(screen.getByRole("button", { name: /click me/i })).toBeInTheDocument();
  });

  it("handles click events", () => {
    const onClick = vi.fn();
    render(<Button onClick={onClick}>Click</Button>);
    fireEvent.click(screen.getByRole("button"));
    expect(onClick).toHaveBeenCalledOnce();
  });

  it("can be disabled", () => {
    render(<Button disabled>Disabled</Button>);
    expect(screen.getByRole("button")).toBeDisabled();
  });

  it("renders primary variant", () => {
    const { container } = render(<Button variant="primary">Primary</Button>);
    const btn = container.firstChild as HTMLElement;
    expect(btn.className).toContain("shadow");
  });

  it("renders different sizes", () => {
    const { container: sm } = render(<Button size="sm">Small</Button>);
    const { container: lg } = render(<Button size="lg">Large</Button>);
    expect((sm.firstChild as HTMLElement).className).toContain("h-7");
    expect((lg.firstChild as HTMLElement).className).toContain("h-11");
  });
});

describe("StatCard", () => {
  it("renders label and value", () => {
    render(<StatCard label="Equity" value="$10,000" />);
    expect(screen.getByText("Equity")).toBeInTheDocument();
    expect(screen.getByText("$10,000")).toBeInTheDocument();
  });

  it("renders change text with positive styling", () => {
    render(<StatCard label="P&L" value="$500" change="+5%" positive={true} />);
    const change = screen.getByText("+5%");
    expect(change.className).toContain("success");
  });

  it("renders change text with negative styling", () => {
    render(<StatCard label="P&L" value="-$200" change="-2%" positive={false} />);
    const change = screen.getByText("-2%");
    expect(change.className).toContain("danger");
  });
});

describe("Progress", () => {
  it("renders with correct aria attributes", () => {
    render(<Progress value={75} />);
    const progressbar = screen.getByRole("progressbar");
    expect(progressbar).toHaveAttribute("aria-valuenow", "75");
    expect(progressbar).toHaveAttribute("aria-valuemin", "0");
    expect(progressbar).toHaveAttribute("aria-valuemax", "100");
  });

  it("clamps value between 0 and 100", () => {
    const { container } = render(<Progress value={150} />);
    const fill = container.querySelector("[style]") as HTMLElement;
    expect(fill.style.width).toBe("100%");
  });

  it("shows label when provided", () => {
    render(<Progress value={50} label="Upload Progress" showValue />);
    expect(screen.getByText("Upload Progress")).toBeInTheDocument();
    expect(screen.getByText("50%")).toBeInTheDocument();
  });
});

describe("Spinner", () => {
  it("renders with status role", () => {
    render(<Spinner />);
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("has accessible label", () => {
    render(<Spinner />);
    expect(screen.getByLabelText("Loading")).toBeInTheDocument();
  });
});

describe("EmptyState", () => {
  it("renders title", () => {
    render(<EmptyState title="No positions" />);
    expect(screen.getByText("No positions")).toBeInTheDocument();
  });

  it("renders description when provided", () => {
    render(<EmptyState title="Empty" description="Start trading to see positions" />);
    expect(screen.getByText("Start trading to see positions")).toBeInTheDocument();
  });

  it("renders action button when provided", () => {
    const onClick = vi.fn();
    render(<EmptyState title="Empty" action={{ label: "Create", onClick }} />);
    const btn = screen.getByRole("button", { name: /create/i });
    fireEvent.click(btn);
    expect(onClick).toHaveBeenCalledOnce();
  });
});

describe("Input", () => {
  it("renders with label", () => {
    render(<Input label="Email" />);
    expect(screen.getByLabelText("Email")).toBeInTheDocument();
  });

  it("shows error message", () => {
    render(<Input label="Email" error="Required field" />);
    expect(screen.getByText("Required field")).toBeInTheDocument();
    expect(screen.getByRole("alert")).toBeInTheDocument();
  });

  it("shows helper text when no error", () => {
    render(<Input label="Email" helperText="Enter your email" />);
    expect(screen.getByText("Enter your email")).toBeInTheDocument();
  });

  it("marks input as invalid when error exists", () => {
    render(<Input label="Email" error="Bad email" />);
    const input = screen.getByLabelText("Email");
    expect(input).toHaveAttribute("aria-invalid", "true");
  });
});

describe("Skeleton", () => {
  it("renders with aria-hidden", () => {
    const { container } = render(<Skeleton className="h-4 w-20" />);
    expect(container.firstChild).toHaveAttribute("aria-hidden", "true");
  });

  it("CardSkeleton renders multiple skeleton elements", () => {
    const { container } = render(<CardSkeleton />);
    const skeletons = container.querySelectorAll("[aria-hidden='true']");
    expect(skeletons.length).toBeGreaterThanOrEqual(3);
  });

  it("TableSkeleton renders configurable rows", () => {
    const { container } = render(<TableSkeleton rows={3} />);
    const skeletons = container.querySelectorAll("[aria-hidden='true']");
    // 1 header + 3 rows = 4
    expect(skeletons.length).toBe(4);
  });
});
