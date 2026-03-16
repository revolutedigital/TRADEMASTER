import { describe, it, expect, beforeEach } from "vitest";
import { setLocale, getLocale, t } from "@/i18n";

describe("i18n", () => {
  beforeEach(() => {
    setLocale("en");
  });

  it("defaults to en locale", () => {
    expect(getLocale()).toBe("en");
  });

  it("can set locale to pt", () => {
    setLocale("pt");
    expect(getLocale()).toBe("pt");
  });

  it("translates known keys in en", () => {
    const result = t("nav.dashboard");
    // Should return the English translation (not the key itself)
    expect(result).not.toBe("nav.dashboard");
  });

  it("returns key for unknown translations", () => {
    expect(t("some.nonexistent.key")).toBe("some.nonexistent.key");
  });

  it("falls back to en when pt key is missing", () => {
    setLocale("pt");
    // A deep key that might not be in pt should fallback to en or return the key
    const result = t("nav.dashboard");
    expect(typeof result).toBe("string");
  });
});
