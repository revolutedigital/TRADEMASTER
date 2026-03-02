import { describe, it, expect } from "vitest";
import { formatCurrency, formatPercent, formatNumber, formatCompactNumber, timeAgo } from "@/lib/utils";

describe("formatCurrency", () => {
  it("formats positive numbers", () => {
    expect(formatCurrency(1234.56)).toBe("$1,234.56");
  });

  it("formats zero", () => {
    expect(formatCurrency(0)).toBe("$0.00");
  });

  it("handles null/undefined", () => {
    expect(formatCurrency(null)).toBe("$0.00");
    expect(formatCurrency(undefined)).toBe("$0.00");
  });

  it("handles NaN/Infinity", () => {
    expect(formatCurrency(NaN)).toBe("$0.00");
    expect(formatCurrency(Infinity)).toBe("$0.00");
  });

  it("respects custom decimals", () => {
    const result = formatCurrency(1234.5678, 4);
    expect(result).toContain("1,234.5678");
  });
});

describe("formatPercent", () => {
  it("formats positive percent with + sign", () => {
    expect(formatPercent(0.1234)).toBe("+12.34%");
  });

  it("formats negative percent without + sign", () => {
    expect(formatPercent(-0.05)).toBe("-5.00%");
  });

  it("handles null/undefined", () => {
    expect(formatPercent(null)).toBe("+0.00%");
    expect(formatPercent(undefined)).toBe("+0.00%");
  });

  it("handles zero", () => {
    expect(formatPercent(0)).toBe("+0.00%");
  });
});

describe("formatNumber", () => {
  it("formats with thousand separators", () => {
    expect(formatNumber(1234567.89)).toBe("1,234,567.89");
  });

  it("formats with custom decimals", () => {
    expect(formatNumber(3.14159, 4)).toContain("3.1416");
  });
});

describe("formatCompactNumber", () => {
  it("formats thousands", () => {
    const result = formatCompactNumber(1500);
    expect(result).toContain("1.5K");
  });

  it("formats millions", () => {
    const result = formatCompactNumber(2500000);
    expect(result).toContain("2.5M");
  });
});

describe("timeAgo", () => {
  it("formats seconds ago", () => {
    const now = new Date();
    const recent = new Date(now.getTime() - 30000); // 30s ago
    expect(timeAgo(recent)).toContain("s ago");
  });

  it("formats minutes ago", () => {
    const now = new Date();
    const recent = new Date(now.getTime() - 120000); // 2m ago
    expect(timeAgo(recent)).toContain("m ago");
  });

  it("formats hours ago", () => {
    const now = new Date();
    const recent = new Date(now.getTime() - 7200000); // 2h ago
    expect(timeAgo(recent)).toContain("h ago");
  });

  it("formats days ago", () => {
    const now = new Date();
    const recent = new Date(now.getTime() - 172800000); // 2d ago
    expect(timeAgo(recent)).toContain("d ago");
  });

  it("accepts string dates", () => {
    const result = timeAgo("2020-01-01T00:00:00Z");
    expect(result).toContain("d ago");
  });
});
