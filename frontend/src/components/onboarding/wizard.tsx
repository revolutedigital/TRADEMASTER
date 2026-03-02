"use client";

import { useOnboardingStore } from "@/stores/onboardingStore";
import {
  Activity,
  Key,
  Shield,
  FlaskConical,
  ChevronRight,
  ChevronLeft,
  X,
} from "lucide-react";

const steps = [
  {
    icon: Activity,
    title: "Welcome to TradeMaster",
    subtitle: "AI-Powered Crypto Trading",
    content: (
      <div className="space-y-4 text-sm text-[var(--color-text-muted)]">
        <p>
          TradeMaster is an AI-powered cryptocurrency trading platform that combines
          machine learning models with real-time market data to generate trading signals
          for BTC and ETH.
        </p>
        <div className="grid grid-cols-2 gap-3">
          <div className="rounded-lg border border-[var(--color-border)] p-3">
            <div className="font-medium text-[var(--color-text)]">Real-Time Data</div>
            <p className="mt-1 text-xs">Live prices from Binance with candlestick charts and technical indicators</p>
          </div>
          <div className="rounded-lg border border-[var(--color-border)] p-3">
            <div className="font-medium text-[var(--color-text)]">ML Signals</div>
            <p className="mt-1 text-xs">LSTM + XGBoost ensemble models generate BUY/HOLD/SELL signals</p>
          </div>
          <div className="rounded-lg border border-[var(--color-border)] p-3">
            <div className="font-medium text-[var(--color-text)]">Paper Trading</div>
            <p className="mt-1 text-xs">Practice with simulated orders before going live</p>
          </div>
          <div className="rounded-lg border border-[var(--color-border)] p-3">
            <div className="font-medium text-[var(--color-text)]">Risk Management</div>
            <p className="mt-1 text-xs">Circuit breakers, position sizing, and drawdown limits</p>
          </div>
        </div>
      </div>
    ),
  },
  {
    icon: Key,
    title: "API Configuration",
    subtitle: "Connect to Binance Testnet",
    content: (
      <div className="space-y-4 text-sm text-[var(--color-text-muted)]">
        <p>
          TradeMaster connects to Binance Testnet by default for safe, risk-free trading practice.
          No real funds are used until you explicitly switch to live mode.
        </p>
        <div className="rounded-lg bg-[var(--color-background)] p-4 space-y-3">
          <div className="flex items-start gap-2">
            <span className="mt-0.5 flex h-5 w-5 items-center justify-center rounded-full bg-[var(--color-primary)]/20 text-xs font-bold text-[var(--color-primary)]">1</span>
            <div>
              <div className="font-medium text-[var(--color-text)]">Create Testnet Account</div>
              <p className="text-xs">Visit testnet.binance.vision and create a free testnet account</p>
            </div>
          </div>
          <div className="flex items-start gap-2">
            <span className="mt-0.5 flex h-5 w-5 items-center justify-center rounded-full bg-[var(--color-primary)]/20 text-xs font-bold text-[var(--color-primary)]">2</span>
            <div>
              <div className="font-medium text-[var(--color-text)]">Generate API Keys</div>
              <p className="text-xs">Generate an API key and secret from the testnet dashboard</p>
            </div>
          </div>
          <div className="flex items-start gap-2">
            <span className="mt-0.5 flex h-5 w-5 items-center justify-center rounded-full bg-[var(--color-primary)]/20 text-xs font-bold text-[var(--color-primary)]">3</span>
            <div>
              <div className="font-medium text-[var(--color-text)]">Configure in Settings</div>
              <p className="text-xs">Go to Settings page and paste your API keys securely</p>
            </div>
          </div>
        </div>
        <p className="text-xs text-yellow-400">
          Your API keys are encrypted at rest and never exposed in the frontend.
        </p>
      </div>
    ),
  },
  {
    icon: Shield,
    title: "Risk Management",
    subtitle: "Understanding Your Safety Net",
    content: (
      <div className="space-y-4 text-sm text-[var(--color-text-muted)]">
        <p>
          TradeMaster includes multiple layers of risk protection to safeguard your capital.
        </p>
        <div className="space-y-3">
          <div className="rounded-lg border border-[var(--color-border)] p-3">
            <div className="font-medium text-[var(--color-text)]">Circuit Breaker</div>
            <p className="mt-1 text-xs">
              Automatically reduces or halts trading when drawdown thresholds are breached.
              States: NORMAL &rarr; REDUCED (-50% size) &rarr; PAUSED &rarr; HALTED
            </p>
          </div>
          <div className="rounded-lg border border-[var(--color-border)] p-3">
            <div className="font-medium text-[var(--color-text)]">Position Limits</div>
            <p className="mt-1 text-xs">
              Maximum 3 concurrent positions with a 40% total portfolio exposure cap.
              Individual position size limited to 20% of equity.
            </p>
          </div>
          <div className="rounded-lg border border-[var(--color-border)] p-3">
            <div className="font-medium text-[var(--color-text)]">Stop Loss / Take Profit</div>
            <p className="mt-1 text-xs">
              Every trade includes automatic stop-loss (2%) and take-profit (4%) levels.
              These are monitored every 5 seconds and executed automatically.
            </p>
          </div>
          <div className="rounded-lg border border-[var(--color-border)] p-3">
            <div className="font-medium text-[var(--color-text)]">Drawdown Limits</div>
            <p className="mt-1 text-xs">
              Daily: -3% | Weekly: -7% | Monthly: -15%.
              Breaching these triggers the circuit breaker for capital preservation.
            </p>
          </div>
        </div>
      </div>
    ),
  },
  {
    icon: FlaskConical,
    title: "Your First Backtest",
    subtitle: "Test Before You Trade",
    content: (
      <div className="space-y-4 text-sm text-[var(--color-text-muted)]">
        <p>
          Before trading with real signals, run a backtest to see how the ML models
          would have performed historically.
        </p>
        <div className="rounded-lg bg-[var(--color-background)] p-4 space-y-3">
          <div className="flex items-start gap-2">
            <span className="mt-0.5 flex h-5 w-5 items-center justify-center rounded-full bg-green-500/20 text-xs font-bold text-green-400">1</span>
            <div>
              <div className="font-medium text-[var(--color-text)]">Navigate to Backtest</div>
              <p className="text-xs">Click &ldquo;Backtest&rdquo; in the sidebar to open the backtesting page</p>
            </div>
          </div>
          <div className="flex items-start gap-2">
            <span className="mt-0.5 flex h-5 w-5 items-center justify-center rounded-full bg-green-500/20 text-xs font-bold text-green-400">2</span>
            <div>
              <div className="font-medium text-[var(--color-text)]">Configure Parameters</div>
              <p className="text-xs">Select BTCUSDT, 1h interval, $10,000 initial capital, and default signal threshold</p>
            </div>
          </div>
          <div className="flex items-start gap-2">
            <span className="mt-0.5 flex h-5 w-5 items-center justify-center rounded-full bg-green-500/20 text-xs font-bold text-green-400">3</span>
            <div>
              <div className="font-medium text-[var(--color-text)]">Analyze Results</div>
              <p className="text-xs">Review equity curve, Sharpe ratio, max drawdown, win rate, and profit factor</p>
            </div>
          </div>
          <div className="flex items-start gap-2">
            <span className="mt-0.5 flex h-5 w-5 items-center justify-center rounded-full bg-green-500/20 text-xs font-bold text-green-400">4</span>
            <div>
              <div className="font-medium text-[var(--color-text)]">Start Paper Trading</div>
              <p className="text-xs">If results look promising, return to Dashboard and start the trading engine</p>
            </div>
          </div>
        </div>
        <p className="mt-2 text-xs text-[var(--color-primary)]">
          Tip: A Sharpe ratio above 1.0 and a profit factor above 1.5 are generally considered good indicators.
        </p>
      </div>
    ),
  },
];

export function OnboardingWizard() {
  const { currentStep, totalSteps, nextStep, prevStep, complete } = useOnboardingStore();
  const step = steps[currentStep];
  const Icon = step.icon;
  const isLast = currentStep === totalSteps - 1;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
      <div className="relative w-full max-w-lg rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] shadow-2xl">
        {/* Close button */}
        <button
          onClick={complete}
          className="absolute right-4 top-4 rounded-md p-1 text-[var(--color-text-muted)] hover:text-[var(--color-text)] transition-colors"
          aria-label="Skip onboarding"
        >
          <X className="h-5 w-5" />
        </button>

        {/* Header */}
        <div className="flex items-center gap-3 border-b border-[var(--color-border)] px-6 py-5">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--color-primary)]/10">
            <Icon className="h-5 w-5 text-[var(--color-primary)]" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-[var(--color-text)]">{step.title}</h2>
            <p className="text-sm text-[var(--color-text-muted)]">{step.subtitle}</p>
          </div>
        </div>

        {/* Content */}
        <div className="px-6 py-5">{step.content}</div>

        {/* Footer */}
        <div className="flex items-center justify-between border-t border-[var(--color-border)] px-6 py-4">
          {/* Progress dots */}
          <div className="flex gap-1.5">
            {Array.from({ length: totalSteps }).map((_, i) => (
              <div
                key={i}
                className={`h-2 w-2 rounded-full transition-colors ${
                  i === currentStep
                    ? "bg-[var(--color-primary)]"
                    : i < currentStep
                    ? "bg-[var(--color-primary)]/40"
                    : "bg-[var(--color-border)]"
                }`}
              />
            ))}
          </div>

          {/* Navigation buttons */}
          <div className="flex gap-2">
            {currentStep > 0 && (
              <button
                onClick={prevStep}
                className="flex items-center gap-1 rounded-lg border border-[var(--color-border)] px-4 py-2 text-sm font-medium text-[var(--color-text-muted)] hover:bg-[var(--color-surface-hover)] transition-colors"
              >
                <ChevronLeft className="h-4 w-4" />
                Back
              </button>
            )}
            <button
              onClick={isLast ? complete : nextStep}
              className="flex items-center gap-1 rounded-lg bg-[var(--color-primary)] px-4 py-2 text-sm font-medium text-white hover:bg-[var(--color-primary)]/90 transition-colors"
            >
              {isLast ? "Get Started" : "Next"}
              {!isLast && <ChevronRight className="h-4 w-4" />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
