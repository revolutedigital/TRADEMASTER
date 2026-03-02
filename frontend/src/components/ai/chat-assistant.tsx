"use client";

import { useState, useRef, useEffect } from "react";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
}

export function AIChatAssistant() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content: "Hi! I'm your TradeMaster AI assistant. Ask me about your portfolio, risk metrics, or request a backtest. Try:\n- \"What's my current risk exposure?\"\n- \"Show my portfolio summary\"\n- \"Run a backtest for SMA crossover on BTCUSDT\"",
      timestamp: Date.now(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMsg: Message = { id: `u-${Date.now()}`, role: "user", content: input.trim(), timestamp: Date.now() };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const response = await processQuery(userMsg.content);
      const assistantMsg: Message = { id: `a-${Date.now()}`, role: "assistant", content: response, timestamp: Date.now() };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch {
      setMessages((prev) => [...prev, { id: `e-${Date.now()}`, role: "assistant", content: "Sorry, I encountered an error. Please try again.", timestamp: Date.now() }]);
    } finally {
      setLoading(false);
    }
  }

  async function processQuery(query: string): Promise<string> {
    const lowerQuery = query.toLowerCase();

    if (lowerQuery.includes("portfolio") || lowerQuery.includes("balance")) {
      try {
        const res = await fetch("/api/v1/portfolio/summary", { credentials: "include" });
        if (res.ok) {
          const data = await res.json();
          return `**Portfolio Summary:**\n- Total Equity: $${Number(data.total_equity).toLocaleString()}\n- Available Balance: $${Number(data.available_balance).toLocaleString()}\n- Unrealized P&L: $${Number(data.unrealized_pnl).toFixed(2)}\n- Daily P&L: $${Number(data.daily_pnl).toFixed(2)}`;
        }
      } catch {}
      return "Unable to fetch portfolio data. Please check your connection.";
    }

    if (lowerQuery.includes("risk") || lowerQuery.includes("exposure") || lowerQuery.includes("drawdown")) {
      try {
        const res = await fetch("/api/v1/risk/dashboard", { credentials: "include" });
        if (res.ok) {
          const data = await res.json();
          return `**Risk Dashboard:**\n- Current Drawdown: ${(Number(data.current_drawdown) * 100).toFixed(2)}%\n- Max Drawdown: ${(Number(data.max_drawdown) * 100).toFixed(2)}%\n- VaR (95%): $${Number(data.var_95).toFixed(2)}\n- Circuit Breaker: ${data.circuit_breaker_state}\n- Open Positions: ${data.open_positions}`;
        }
      } catch {}
      return "Unable to fetch risk data. The risk service may be unavailable.";
    }

    if (lowerQuery.includes("price") || lowerQuery.includes("btc") || lowerQuery.includes("eth")) {
      try {
        const res = await fetch("/api/v1/market/tickers", { credentials: "include" });
        if (res.ok) {
          const tickers = await res.json();
          const lines = tickers.map((t: Record<string, unknown>) => `- ${t.symbol}: $${Number(t.price).toLocaleString()} (${(Number(t.change_24h) * 100).toFixed(2)}%)`);
          return `**Current Prices:**\n${lines.join("\n")}`;
        }
      } catch {}
      return "Unable to fetch market data.";
    }

    if (lowerQuery.includes("trade") || lowerQuery.includes("history") || lowerQuery.includes("recent")) {
      try {
        const res = await fetch("/api/v1/trading/history?limit=5", { credentials: "include" });
        if (res.ok) {
          const trades = await res.json();
          if (trades.length === 0) return "No recent trades found.";
          const lines = trades.map((t: Record<string, unknown>) => `- ${t.side} ${t.symbol}: ${t.quantity} @ $${Number(t.price).toLocaleString()} (${t.status})`);
          return `**Recent Trades:**\n${lines.join("\n")}`;
        }
      } catch {}
      return "Unable to fetch trade history.";
    }

    if (lowerQuery.includes("help") || lowerQuery.includes("what can you do")) {
      return `I can help you with:\n\n**Portfolio:** "Show my portfolio summary", "What's my balance?"\n**Risk:** "What's my risk exposure?", "Show drawdown"\n**Market:** "What's the BTC price?", "Show current prices"\n**Trades:** "Show recent trades", "Trade history"\n**Analysis:** "Run a backtest", "Show signals"\n\nJust ask naturally and I'll fetch the data for you!`;
    }

    return "I understand you're asking about: \"" + query + "\". Try asking about portfolio, risk, prices, or trades. Type 'help' for a list of commands.";
  }

  return (
    <>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="fixed bottom-6 left-6 w-12 h-12 bg-purple-600 hover:bg-purple-700 rounded-full flex items-center justify-center text-white shadow-lg z-40 transition-colors"
        title="AI Assistant"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="10" />
          <path d="M8 14s1.5 2 4 2 4-2 4-2" />
          <line x1="9" y1="9" x2="9.01" y2="9" />
          <line x1="15" y1="9" x2="15.01" y2="9" />
        </svg>
      </button>

      {isOpen && (
        <div className="fixed bottom-20 left-6 w-96 h-[500px] bg-[#141922] border border-gray-700 rounded-xl shadow-2xl flex flex-col z-40">
          <div className="p-3 border-b border-gray-700 flex justify-between items-center bg-purple-600/10">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-green-500" />
              <h3 className="text-white font-semibold text-sm">AI Assistant</h3>
            </div>
            <button onClick={() => setIsOpen(false)} className="text-gray-400 hover:text-white">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M18 6L6 18M6 6l12 12" />
              </svg>
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map((msg) => (
              <div key={msg.id} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                <div className={`max-w-[80%] rounded-lg px-3 py-2 text-sm whitespace-pre-line ${msg.role === "user" ? "bg-blue-600 text-white" : "bg-[#1a1f2e] text-gray-300"}`}>
                  {msg.content}
                </div>
              </div>
            ))}
            {loading && (
              <div className="flex justify-start">
                <div className="bg-[#1a1f2e] rounded-lg px-3 py-2 text-sm text-gray-400">
                  <span className="animate-pulse">Thinking...</span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <form onSubmit={handleSubmit} className="p-3 border-t border-gray-700">
            <div className="flex gap-2">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask me anything..."
                className="flex-1 bg-[#1a1f2e] text-white text-sm rounded-lg px-3 py-2 border border-gray-600 focus:border-purple-500 outline-none"
                disabled={loading}
              />
              <button type="submit" disabled={loading} className="px-3 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 text-white rounded-lg text-sm transition-colors">
                Send
              </button>
            </div>
          </form>
        </div>
      )}
    </>
  );
}
