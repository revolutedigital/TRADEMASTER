import { describe, it, expect } from "vitest";
import { validateLogin, validatePaperOrder, validateBacktest } from "@/lib/schemas";

describe("validateLogin", () => {
  it("returns no errors for valid credentials", () => {
    const errors = validateLogin({ username: "admin", password: "password123" });
    expect(errors).toEqual([]);
  });

  it("returns error for short username", () => {
    const errors = validateLogin({ username: "ab", password: "password123" });
    expect(errors).toContain("Username must be at least 3 characters");
  });

  it("returns error for empty username", () => {
    const errors = validateLogin({ username: "", password: "password123" });
    expect(errors.length).toBeGreaterThan(0);
  });

  it("returns error for short password", () => {
    const errors = validateLogin({ username: "admin", password: "12345" });
    expect(errors).toContain("Password must be at least 6 characters");
  });

  it("returns multiple errors for invalid credentials", () => {
    const errors = validateLogin({ username: "a", password: "12" });
    expect(errors.length).toBe(2);
  });
});

describe("validatePaperOrder", () => {
  it("returns no errors for valid order", () => {
    const errors = validatePaperOrder({
      symbol: "BTCUSDT",
      side: "BUY",
      quantity: 0.01,
      stop_loss_pct: null,
      take_profit_pct: null,
    });
    expect(errors).toEqual([]);
  });

  it("returns error for invalid symbol", () => {
    const errors = validatePaperOrder({
      symbol: "DOGEUSDT" as "BTCUSDT",
      side: "BUY",
      quantity: 0.01,
      stop_loss_pct: null,
      take_profit_pct: null,
    });
    expect(errors).toContain("Symbol must be BTCUSDT or ETHUSDT");
  });

  it("returns error for invalid side", () => {
    const errors = validatePaperOrder({
      symbol: "BTCUSDT",
      side: "HOLD" as "BUY",
      quantity: 0.01,
      stop_loss_pct: null,
      take_profit_pct: null,
    });
    expect(errors).toContain("Side must be BUY or SELL");
  });

  it("returns error for zero quantity", () => {
    const errors = validatePaperOrder({
      symbol: "BTCUSDT",
      side: "BUY",
      quantity: 0,
      stop_loss_pct: null,
      take_profit_pct: null,
    });
    expect(errors).toContain("Quantity must be positive");
  });

  it("returns error for negative quantity", () => {
    const errors = validatePaperOrder({
      symbol: "ETHUSDT",
      side: "SELL",
      quantity: -0.5,
      stop_loss_pct: null,
      take_profit_pct: null,
    });
    expect(errors).toContain("Quantity must be positive");
  });
});

describe("validateBacktest", () => {
  const validData = {
    symbol: "BTCUSDT",
    interval: "1h",
    initial_capital: 10000,
    signal_threshold: 0.5,
    atr_stop_multiplier: 2.0,
    risk_reward_ratio: 2.0,
  };

  it("returns no errors for valid data", () => {
    expect(validateBacktest(validData)).toEqual([]);
  });

  it("returns error for capital too low", () => {
    const errors = validateBacktest({ ...validData, initial_capital: 50 });
    expect(errors.some((e) => e.includes("Initial capital"))).toBe(true);
  });

  it("returns error for capital too high", () => {
    const errors = validateBacktest({ ...validData, initial_capital: 2_000_000 });
    expect(errors.some((e) => e.includes("Initial capital"))).toBe(true);
  });

  it("returns error for signal threshold out of range", () => {
    const errors = validateBacktest({ ...validData, signal_threshold: 0.05 });
    expect(errors.some((e) => e.includes("Signal threshold"))).toBe(true);
  });

  it("returns error for ATR multiplier out of range", () => {
    const errors = validateBacktest({ ...validData, atr_stop_multiplier: 0.1 });
    expect(errors.some((e) => e.includes("ATR stop multiplier"))).toBe(true);
  });

  it("returns error for risk/reward ratio out of range", () => {
    const errors = validateBacktest({ ...validData, risk_reward_ratio: 15 });
    expect(errors.some((e) => e.includes("Risk/reward"))).toBe(true);
  });
});
