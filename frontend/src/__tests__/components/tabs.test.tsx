import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { Tabs, TabPanel } from "@/components/ui/tabs";

const tabs = [
  { id: "a", label: "Tab A" },
  { id: "b", label: "Tab B" },
  { id: "c", label: "Tab C" },
];

describe("Tabs (underline variant)", () => {
  it("renders all tab buttons with role=tab", () => {
    render(<Tabs tabs={tabs} activeTab="a" onChange={() => {}} />);
    const tabEls = screen.getAllByRole("tab");
    expect(tabEls).toHaveLength(3);
    expect(tabEls[0]).toHaveTextContent("Tab A");
  });

  it("marks active tab with aria-selected=true", () => {
    render(<Tabs tabs={tabs} activeTab="b" onChange={() => {}} />);
    const tabEls = screen.getAllByRole("tab");
    expect(tabEls[1]).toHaveAttribute("aria-selected", "true");
    expect(tabEls[0]).toHaveAttribute("aria-selected", "false");
  });

  it("calls onChange when a tab is clicked", () => {
    const onChange = vi.fn();
    render(<Tabs tabs={tabs} activeTab="a" onChange={onChange} />);
    fireEvent.click(screen.getByText("Tab C"));
    expect(onChange).toHaveBeenCalledWith("c");
  });

  it("navigates with ArrowRight key", () => {
    const onChange = vi.fn();
    render(<Tabs tabs={tabs} activeTab="a" onChange={onChange} />);
    const firstTab = screen.getAllByRole("tab")[0];
    fireEvent.keyDown(firstTab, { key: "ArrowRight" });
    expect(onChange).toHaveBeenCalledWith("b");
  });

  it("navigates with ArrowLeft key (wraps around)", () => {
    const onChange = vi.fn();
    render(<Tabs tabs={tabs} activeTab="a" onChange={onChange} />);
    const firstTab = screen.getAllByRole("tab")[0];
    fireEvent.keyDown(firstTab, { key: "ArrowLeft" });
    expect(onChange).toHaveBeenCalledWith("c");
  });
});

describe("Tabs (pills variant)", () => {
  it("renders tablist with pills styling", () => {
    render(<Tabs tabs={tabs} activeTab="a" onChange={() => {}} variant="pills" />);
    expect(screen.getByRole("tablist")).toBeInTheDocument();
    expect(screen.getAllByRole("tab")).toHaveLength(3);
  });
});

describe("TabPanel", () => {
  it("renders children when active", () => {
    render(<TabPanel id="a" activeTab="a">Content A</TabPanel>);
    expect(screen.getByText("Content A")).toBeInTheDocument();
  });

  it("returns null when not active", () => {
    const { container } = render(<TabPanel id="b" activeTab="a">Content B</TabPanel>);
    expect(container.innerHTML).toBe("");
  });

  it("has tabpanel role", () => {
    render(<TabPanel id="a" activeTab="a">Content</TabPanel>);
    expect(screen.getByRole("tabpanel")).toBeInTheDocument();
  });
});
