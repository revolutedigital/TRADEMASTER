"use client";

import { useState, useEffect, useCallback } from "react";
import { Sidebar } from "@/components/ui/sidebar";

interface JournalEntry {
  id: string;
  trade_id: string | null;
  notes: string;
  tags: string[];
  sentiment: "bullish" | "bearish" | "neutral";
  lessons_learned: string;
  created_at: string;
}

export default function JournalPage() {
  const [entries, setEntries] = useState<JournalEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState<{ notes: string; tags: string; sentiment: "bullish" | "bearish" | "neutral"; lessons_learned: string }>({ notes: "", tags: "", sentiment: "neutral", lessons_learned: "" });

  const fetchEntries = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/journal", { credentials: "include" });
      if (res.ok) setEntries(await res.json());
    } catch {} finally { setLoading(false); }
  }, []);

  useEffect(() => { fetchEntries(); }, [fetchEntries]);

  async function createEntry(e: React.FormEvent) {
    e.preventDefault();
    try {
      const res = await fetch("/api/v1/journal", {
        method: "POST", credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...form, tags: form.tags.split(",").map((t) => t.trim()).filter(Boolean) }),
      });
      if (res.ok) { setShowForm(false); setForm({ notes: "", tags: "", sentiment: "neutral", lessons_learned: "" }); fetchEntries(); }
    } catch {}
  }

  const sentimentColors = { bullish: "text-green-400 bg-green-500/20", bearish: "text-red-400 bg-red-500/20", neutral: "text-yellow-400 bg-yellow-500/20" };

  return (
    <div className="flex h-screen bg-[#0a0e17]">
      <Sidebar />
      <main className="flex-1 overflow-y-auto p-6">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-bold text-white">Trading Journal</h1>
          <button onClick={() => setShowForm(!showForm)} className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors">
            {showForm ? "Cancel" : "+ New Entry"}
          </button>
        </div>

        {showForm && (
          <form onSubmit={createEntry} className="bg-[#141922] rounded-xl p-6 mb-6 space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Notes</label>
              <textarea rows={4} value={form.notes} onChange={(e) => setForm({ ...form, notes: e.target.value })} className="w-full bg-[#1a1f2e] text-white rounded-lg px-3 py-2 border border-gray-700 resize-none" required />
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm text-gray-400 mb-1">Sentiment</label>
                <select value={form.sentiment} onChange={(e) => setForm({ ...form, sentiment: e.target.value as "bullish" | "bearish" | "neutral" })} className="w-full bg-[#1a1f2e] text-white rounded-lg px-3 py-2 border border-gray-700">
                  <option value="bullish">Bullish</option>
                  <option value="bearish">Bearish</option>
                  <option value="neutral">Neutral</option>
                </select>
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-1">Tags (comma-separated)</label>
                <input value={form.tags} onChange={(e) => setForm({ ...form, tags: e.target.value })} className="w-full bg-[#1a1f2e] text-white rounded-lg px-3 py-2 border border-gray-700" placeholder="swing, btc, breakout" />
              </div>
              <div className="flex items-end">
                <button type="submit" className="w-full px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium transition-colors">Save Entry</button>
              </div>
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">Lessons Learned</label>
              <textarea rows={2} value={form.lessons_learned} onChange={(e) => setForm({ ...form, lessons_learned: e.target.value })} className="w-full bg-[#1a1f2e] text-white rounded-lg px-3 py-2 border border-gray-700 resize-none" />
            </div>
          </form>
        )}

        {loading ? (
          <div className="space-y-4">{[...Array(3)].map((_, i) => <div key={i} className="bg-[#141922] rounded-xl p-6 animate-pulse h-32" />)}</div>
        ) : entries.length === 0 ? (
          <div className="bg-[#141922] rounded-xl p-12 text-center">
            <p className="text-gray-400 text-lg">No journal entries yet</p>
            <p className="text-gray-500 text-sm mt-2">Start documenting your trades and lessons learned.</p>
          </div>
        ) : (
          <div className="space-y-4">
            {entries.map((entry) => (
              <div key={entry.id} className="bg-[#141922] rounded-xl p-6">
                <div className="flex justify-between items-start mb-3">
                  <div className="flex items-center gap-2">
                    <span className={`text-xs px-2 py-1 rounded ${sentimentColors[entry.sentiment]}`}>{entry.sentiment}</span>
                    {entry.tags.map((tag) => (
                      <span key={tag} className="text-xs bg-gray-700 text-gray-300 px-2 py-1 rounded">#{tag}</span>
                    ))}
                  </div>
                  <span className="text-xs text-gray-500">{new Date(entry.created_at).toLocaleDateString()}</span>
                </div>
                <p className="text-white mb-2">{entry.notes}</p>
                {entry.lessons_learned && (
                  <div className="mt-3 p-3 bg-[#1a1f2e] rounded-lg">
                    <span className="text-xs text-gray-400">Lessons Learned</span>
                    <p className="text-sm text-gray-300 mt-1">{entry.lessons_learned}</p>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
