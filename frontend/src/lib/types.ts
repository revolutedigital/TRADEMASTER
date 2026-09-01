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
  execution_mode: "PAPER" | "TESTNET" | "LIVE";
  protective_order_list_id: number | null;
  protective_quantity: number | null;
  protection_status: "LOCAL" | "ACTIVE" | "MISSING" | "EXITING" | "EXIT_FILLED";
  opened_at: string;
}

export interface Order {
  id: string;
  symbol: string;
  side: "BUY" | "SELL";
  order_type: "MARKET" | "LIMIT";
  status: "PENDING" | "SUBMITTED" | "PARTIAL" | "FILLED" | "CANCELLED";
  execution_mode: "PAPER" | "TESTNET" | "LIVE";
  protective_order_list_id: number | null;
  protective_quantity: number | null;
  price: number;
  quantity: number;
  filled_quantity: number;
  avg_fill_price: number | null;
  commission: number;
  created_at: string;
}

export interface Signal {
  id: number;
  symbol: string;
  action: "BUY" | "HOLD" | "SELL";
  strength: number;
  confidence: number;
  model_source: string;
  timeframe: string;
  was_executed: boolean;
  evidence: SignalEvidence | null;
  generated_at: string;
}

export interface SignalEvidence {
  signal_source: string;
  strategy_deployment_id: number | null;
  signal_threshold: number;
  agreement_ratio: number;
  votes: SignalVote[];
  regime: SignalRegime;
  price: number;
  atr: number;
  atr_pct: number;
}

export interface SignalVote {
  model: string;
  action: "BUY" | "HOLD" | "SELL";
  score: number;
  confidence: number;
}

export interface SignalRegime {
  market: string;
  volatility: string;
  confidence: number;
  position_size_multiplier: number;
}

export interface PortfolioSummary {
  execution_mode: "PAPER" | "TESTNET" | "LIVE";
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
  id: number | null;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  total_return: number;
  total_return_pct: number;
  profit_factor: number;
  sharpe_ratio: number;
  max_drawdown: number;
  max_drawdown_pct: number;
  expectancy: number;
  equity_curve: number[];
  strategy?: {
    name: string;
    execution_profile: "spot_long_only" | "model_long_short";
    research_only: boolean;
    indicators: string[];
    min_confirmations: number | null;
  } | null;
}

export interface StrategyDeployment {
  id: number;
  source_backtest_id: number;
  symbol: string;
  interval: string;
  target_execution_mode: "PAPER" | "TESTNET" | "LIVE";
  status: "APPROVED" | "ACTIVE" | "REJECTED" | "DISABLED";
  total_test_trades: number;
  walk_forward_windows: number;
  avg_return_pct: number;
  avg_sharpe: number;
  avg_max_drawdown_pct: number;
  avg_profit_factor: number;
  consistency_score: number;
  overfitting_score: number;
  rejection_reason: string | null;
  activated_at: string | null;
  created_at: string;
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
