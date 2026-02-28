/** Shared TypeScript types for the TradeMaster dashboard. */

export interface Kline {
  open_time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface TickerPrice {
  symbol: string;
  price: number;
  change_24h: number;
  volume_24h: number;
  high_24h: number;
  low_24h: number;
}

export interface Position {
  id: string;
  symbol: string;
  side: "LONG" | "SHORT";
  entry_price: number;
  quantity: number;
  current_price: number;
  unrealized_pnl: number;
  realized_pnl: number;
  stop_loss_price: number | null;
  take_profit_price: number | null;
  opened_at: string;
}

export interface Order {
  id: string;
  symbol: string;
  side: "BUY" | "SELL";
  type: "MARKET" | "LIMIT";
  status: "PENDING" | "SUBMITTED" | "PARTIAL" | "FILLED" | "CANCELLED";
  price: number;
  quantity: number;
  filled_qty: number;
  created_at: string;
}

export interface Signal {
  id: string;
  symbol: string;
  action: "BUY" | "HOLD" | "SELL";
  strength: number;
  confidence: number;
  model_source: string;
  created_at: string;
}

export interface PortfolioSummary {
  total_equity: number;
  available_balance: number;
  total_unrealized_pnl: number;
  total_realized_pnl: number;
  total_exposure: number;
  exposure_pct: number;
  open_positions: number;
  daily_pnl: number;
  daily_pnl_pct: number;
}

export interface RiskStatus {
  state: "NORMAL" | "REDUCED" | "PAUSED" | "HALTED";
  circuit_breaker_state: "NORMAL" | "REDUCED" | "PAUSED" | "HALTED";
  can_trade: boolean;
  position_size_multiplier: number;
  daily_drawdown: number;
  weekly_drawdown: number;
  monthly_drawdown: number;
  max_drawdown: number;
  peak_equity: number;
}

export interface BacktestRequest {
  symbol: string;
  interval: string;
  initial_capital: number;
  signal_threshold: number;
  start_date?: string;
  end_date?: string;
}

export interface BacktestResult {
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  total_return: number;
  total_return_pct: number;
  profit_factor: number;
  sharpe_ratio: number;
  max_drawdown: number;
  equity_curve: number[];
}

export interface SystemHealth {
  status: string;
  version: string;
  uptime: number;
  services: Record<string, string>;
}

export type TimeInterval = "1m" | "5m" | "15m" | "1h" | "4h" | "1d";

// WebSocket message types
export interface WSMessage {
  type: "kline" | "ticker" | "signal" | "order" | "position" | "risk_status";
  data: Record<string, unknown>;
}
