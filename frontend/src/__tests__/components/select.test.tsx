import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { Select } from "@/components/ui/select";

const options = [
  { value: "btc", label: "Bitcoin" },
  { value: "eth", label: "Ethereum" },
  { value: "sol", label: "Solana" },
];

describe("Select", () => {
  it("renders combobox role with placeholder when no value", () => {
    render(<Select options={options} value="" onChange={() => {}} />);
    expect(screen.getByRole("combobox")).toBeInTheDocument();
    expect(screen.getByText("Select...")).toBeInTheDocument();
  });

  it("displays the selected option label", () => {
    render(<Select options={options} value="eth" onChange={() => {}} />);
    expect(screen.getByText("Ethereum")).toBeInTheDocument();
  });

  it("opens dropdown on click and shows options", () => {
    render(<Select options={options} value="" onChange={() => {}} />);
    fireEvent.click(screen.getByRole("combobox"));
    expect(screen.getByRole("listbox")).toBeInTheDocument();
    expect(screen.getAllByRole("option")).toHaveLength(3);
  });

  it("calls onChange when an option is clicked", () => {
    const onChange = vi.fn();
    render(<Select options={options} value="" onChange={onChange} />);
    fireEvent.click(screen.getByRole("combobox"));
    fireEvent.click(screen.getByText("Solana"));
    expect(onChange).toHaveBeenCalledWith("sol");
  });

  it("renders label when provided", () => {
    render(<Select options={options} value="" onChange={() => {}} label="Coin" />);
    expect(screen.getByText("Coin")).toBeInTheDocument();
  });

  it("renders error message when provided", () => {
    render(<Select options={options} value="" onChange={() => {}} error="Required" />);
    expect(screen.getByRole("alert")).toHaveTextContent("Required");
  });

  it("opens with ArrowDown key", () => {
    // scrollIntoView is not available in jsdom
    Element.prototype.scrollIntoView = vi.fn();
    render(<Select options={options} value="" onChange={() => {}} />);
    fireEvent.keyDown(screen.getByRole("combobox"), { key: "ArrowDown" });
    expect(screen.getByRole("listbox")).toBeInTheDocument();
  });

  it("closes with Escape key", () => {
    render(<Select options={options} value="" onChange={() => {}} />);
    fireEvent.click(screen.getByRole("combobox"));
    expect(screen.getByRole("listbox")).toBeInTheDocument();
    fireEvent.keyDown(screen.getByRole("combobox"), { key: "Escape" });
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
  });

  it("does not open when disabled", () => {
    render(<Select options={options} value="" onChange={() => {}} disabled />);
    fireEvent.click(screen.getByRole("combobox"));
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
  });
});
