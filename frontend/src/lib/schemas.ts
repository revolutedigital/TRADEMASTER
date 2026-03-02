/**
 * Zod validation schemas for forms.
 * Used with react-hook-form for client-side validation.
 */

// Using simple validation since zod may not be installed yet.
// These provide type-safe validation without the dependency.

export interface LoginFormData {
  username: string;
  password: string;
}

export interface PaperOrderFormData {
  symbol: "BTCUSDT" | "ETHUSDT";
  side: "BUY" | "SELL";
  quantity: number;
  stop_loss_pct: number | null;
  take_profit_pct: number | null;
}

export interface BacktestFormData {
  symbol: string;
  interval: string;
  initial_capital: number;
  signal_threshold: number;
  atr_stop_multiplier: number;
  risk_reward_ratio: number;
}

export function validateLogin(data: LoginFormData): string[] {
  const errors: string[] = [];
  if (!data.username || data.username.length < 3) {
    errors.push("Username must be at least 3 characters");
  }
  if (!data.password || data.password.length < 6) {
    errors.push("Password must be at least 6 characters");
  }
  return errors;
}

export function validatePaperOrder(data: PaperOrderFormData): string[] {
  const errors: string[] = [];
  if (!["BTCUSDT", "ETHUSDT"].includes(data.symbol)) {
    errors.push("Symbol must be BTCUSDT or ETHUSDT");
  }
  if (!["BUY", "SELL"].includes(data.side)) {
    errors.push("Side must be BUY or SELL");
  }
  if (data.quantity <= 0) {
    errors.push("Quantity must be positive");
  }
  return errors;
}

export function validateBacktest(data: BacktestFormData): string[] {
  const errors: string[] = [];
  if (data.initial_capital < 100 || data.initial_capital > 1_000_000) {
    errors.push("Initial capital must be between $100 and $1,000,000");
  }
  if (data.signal_threshold < 0.1 || data.signal_threshold > 0.9) {
    errors.push("Signal threshold must be between 0.1 and 0.9");
  }
  if (data.atr_stop_multiplier < 0.5 || data.atr_stop_multiplier > 5.0) {
    errors.push("ATR stop multiplier must be between 0.5 and 5.0");
  }
  if (data.risk_reward_ratio < 0.5 || data.risk_reward_ratio > 10.0) {
    errors.push("Risk/reward ratio must be between 0.5 and 10.0");
  }
  return errors;
}
