import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { ExportButton } from "@/components/ui/export-button";

describe("ExportButton", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders with default label", () => {
    render(<ExportButton endpoint="/api/export" filename="data.csv" />);
    expect(screen.getByLabelText("Exportar CSV")).toBeInTheDocument();
    expect(screen.getByText("Exportar CSV")).toBeInTheDocument();
  });

  it("renders with custom label", () => {
    render(<ExportButton endpoint="/api/export" filename="data.csv" label="Download" />);
    expect(screen.getByText("Download")).toBeInTheDocument();
  });

  it("shows Exportando... while loading", async () => {
    // Mock fetch to hang
    vi.spyOn(globalThis, "fetch").mockImplementation(() => new Promise(() => {}));
    render(<ExportButton endpoint="/api/export" filename="data.csv" />);
    fireEvent.click(screen.getByText("Exportar CSV"));
    await waitFor(() => expect(screen.getByText("Exportando...")).toBeInTheDocument());
  });

  it("has an svg icon", () => {
    const { container } = render(<ExportButton endpoint="/api/export" filename="data.csv" />);
    expect(container.querySelector("svg")).toBeInTheDocument();
  });
});
