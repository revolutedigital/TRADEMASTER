"use client";

import { useState } from "react";
import { useAuthStore } from "@/stores/authStore";
import { Logo } from "@/components/ui/logo";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/progress";
import { Eye, EyeOff, Lock, User, TrendingUp, Shield, Brain, Zap } from "lucide-react";
import { cn } from "@/lib/utils";

const features = [
  { icon: TrendingUp, label: "AI-Powered Signals", desc: "LSTM & XGBoost models for BTC/ETH" },
  { icon: Shield, label: "Risk Management", desc: "Circuit breaker, VaR, position sizing" },
  { icon: Brain, label: "Machine Learning", desc: "Real-time model inference & drift detection" },
  { icon: Zap, label: "Paper Trading", desc: "Test strategies without risking capital" },
];

export function LoginPage() {
  const { login, isLoading, error } = useAuthStore();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [shakeError, setShakeError] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await login(username, password);
    if (error) {
      setShakeError(true);
      setTimeout(() => setShakeError(false), 300);
    }
  };

  return (
    <div className="flex min-h-screen">
      {/* Left Panel — Branding */}
      <div className="hidden lg:flex lg:w-1/2 relative overflow-hidden flex-col justify-between p-12 bg-gradient-to-br from-[#0c0c1d] via-[#111128] to-[#0a0a1a]">
        {/* Gradient orbs */}
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-[var(--color-primary)] rounded-full opacity-[0.07] blur-[120px]" aria-hidden="true" />
        <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-purple-500 rounded-full opacity-[0.05] blur-[100px]" aria-hidden="true" />

        <div className="relative z-10">
          <Logo size="lg" />
          <p className="mt-4 text-lg text-[var(--color-text-muted)] max-w-md">
            AI-powered cryptocurrency trading platform for institutional-grade performance.
          </p>
        </div>

        <div className="relative z-10 space-y-4">
          {features.map(({ icon: Icon, label, desc }) => (
            <div key={label} className="flex items-start gap-4">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-[var(--color-primary-light)] text-[var(--color-primary)]">
                <Icon className="h-5 w-5" />
              </div>
              <div>
                <p className="text-sm font-medium">{label}</p>
                <p className="text-xs text-[var(--color-text-muted)]">{desc}</p>
              </div>
            </div>
          ))}
        </div>

        <p className="relative z-10 text-xs text-[var(--color-text-faint)]">
          &copy; {new Date().getFullYear()} TradeMaster. Paper trading mode.
        </p>
      </div>

      {/* Right Panel — Form */}
      <div className="flex flex-1 items-center justify-center bg-[var(--color-background)] p-6 sm:p-12">
        <div className={cn("w-full max-w-sm space-y-8", shakeError && "animate-shake")}>
          {/* Mobile logo */}
          <div className="lg:hidden flex justify-center mb-4">
            <Logo size="lg" />
          </div>

          <div className="text-center lg:text-left">
            <h1 className="text-2xl font-bold tracking-tight">Welcome back</h1>
            <p className="mt-1 text-sm text-[var(--color-text-muted)]">
              Sign in to access your trading dashboard
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-5">
            <Input
              label="Username"
              id="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              leftIcon={<User className="h-4 w-4" />}
              placeholder="admin"
              required
              autoFocus
              autoComplete="username"
            />

            <div>
              <Input
                label="Password"
                id="password"
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                leftIcon={<Lock className="h-4 w-4" />}
                rightIcon={
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="hover:text-[var(--color-text)] transition-colors"
                    aria-label={showPassword ? "Hide password" : "Show password"}
                    tabIndex={-1}
                  >
                    {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                }
                placeholder="Enter password"
                required
                autoComplete="current-password"
              />
            </div>

            {error && (
              <div className="rounded-lg bg-[var(--color-danger-light)] border border-[var(--color-danger)]/20 px-3 py-2 text-sm text-[var(--color-danger)] animate-fade-in" role="alert">
                {error}
              </div>
            )}

            <Button
              type="submit"
              variant="primary"
              size="lg"
              disabled={isLoading}
              className="w-full relative"
            >
              {isLoading ? (
                <span className="flex items-center gap-2">
                  <Spinner size="sm" className="text-white" />
                  Signing in...
                </span>
              ) : (
                "Sign In"
              )}
            </Button>
          </form>

          <div className="flex items-center justify-center gap-2 pt-2">
            <div className="h-1.5 w-1.5 rounded-full bg-[var(--color-success)]" />
            <p className="text-xs text-[var(--color-text-faint)]">
              Paper Trading Mode — No real funds at risk
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
