"use client";

import { useState, useEffect, useCallback } from "react";
import { BookOpen, Plus } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { PageHeader } from "@/components/ui/page-header";
import { EmptyState } from "@/components/ui/empty-state";
import { Spinner } from "@/components/ui/progress";
import { apiFetch } from "@/lib/utils";

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
      const data = await apiFetch<JournalEntry[]>("/api/v1/journal");
      setEntries(data);
    } catch {} finally { setLoading(false); }
  }, []);

  useEffect(() => { fetchEntries(); }, [fetchEntries]);

  async function createEntry(e: React.FormEvent) {
    e.preventDefault();
    try {
      await apiFetch("/api/v1/journal", {
        method: "POST",
        body: JSON.stringify({ ...form, tags: form.tags.split(",").map((t) => t.trim()).filter(Boolean) }),
      });
      setShowForm(false); setForm({ notes: "", tags: "", sentiment: "neutral", lessons_learned: "" }); fetchEntries();
    } catch {}
  }

  const sentimentStyles = {
    bullish: "bg-[var(--color-success)]/15 text-[var(--color-success)]",
    bearish: "bg-[var(--color-danger)]/15 text-[var(--color-danger)]",
    neutral: "bg-[var(--color-warning)]/15 text-[var(--color-warning)]",
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Trading Journal"
        description="Document your trades, lessons, and insights"
        actions={
          <Button variant={showForm ? "ghost" : "primary"} size="sm" onClick={() => setShowForm(!showForm)}>
            {showForm ? "Cancel" : <><Plus className="mr-1.5 h-4 w-4" /> New Entry</>}
          </Button>
        }
      />

      {showForm && (
        <Card>
          <CardContent>
            <form onSubmit={createEntry} className="space-y-4">
              <div>
                <label className="block text-xs font-medium uppercase tracking-wider text-[var(--color-text-muted)] mb-1.5">Notes</label>
                <textarea
                  rows={4}
                  value={form.notes}
                  onChange={(e) => setForm({ ...form, notes: e.target.value })}
                  className="w-full rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm text-[var(--color-text)] resize-none focus:border-[var(--color-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20"
                  required
                />
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="block text-xs font-medium uppercase tracking-wider text-[var(--color-text-muted)] mb-1.5">Sentiment</label>
                  <select
                    value={form.sentiment}
                    onChange={(e) => setForm({ ...form, sentiment: e.target.value as "bullish" | "bearish" | "neutral" })}
                    className="w-full rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm text-[var(--color-text)] focus:border-[var(--color-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20"
                  >
                    <option value="bullish">Bullish</option>
                    <option value="bearish">Bearish</option>
                    <option value="neutral">Neutral</option>
                  </select>
                </div>
                <Input
                  label="Tags (comma-separated)"
                  value={form.tags}
                  onChange={(e) => setForm({ ...form, tags: e.target.value })}
                  placeholder="swing, btc, breakout"
                />
                <div className="flex items-end">
                  <Button type="submit" variant="success" className="w-full">Save Entry</Button>
                </div>
              </div>
              <div>
                <label className="block text-xs font-medium uppercase tracking-wider text-[var(--color-text-muted)] mb-1.5">Lessons Learned</label>
                <textarea
                  rows={2}
                  value={form.lessons_learned}
                  onChange={(e) => setForm({ ...form, lessons_learned: e.target.value })}
                  className="w-full rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm text-[var(--color-text)] resize-none focus:border-[var(--color-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20"
                />
              </div>
            </form>
          </CardContent>
        </Card>
      )}

      {loading ? (
        <div className="flex justify-center py-12"><Spinner size="lg" /></div>
      ) : entries.length === 0 ? (
        <Card>
          <EmptyState
            icon={<BookOpen className="h-7 w-7" />}
            title="No journal entries yet"
            description="Start documenting your trades and lessons learned."
            action={{ label: "+ New Entry", onClick: () => setShowForm(true) }}
          />
        </Card>
      ) : (
        <div className="space-y-4">
          {entries.map((entry) => (
            <Card key={entry.id} className="interactive">
              <CardContent>
                <div className="flex justify-between items-start mb-3">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className={`text-xs px-2.5 py-1 rounded-full font-medium ${sentimentStyles[entry.sentiment]}`}>
                      {entry.sentiment}
                    </span>
                    {entry.tags.map((tag) => (
                      <span key={tag} className="text-xs rounded-full bg-[var(--color-surface-hover)] px-2.5 py-1 text-[var(--color-text-muted)]">
                        #{tag}
                      </span>
                    ))}
                  </div>
                  <span className="text-xs text-[var(--color-text-muted)] tabular-nums">
                    {new Date(entry.created_at).toLocaleDateString()}
                  </span>
                </div>
                <p className="text-[var(--color-text)]">{entry.notes}</p>
                {entry.lessons_learned && (
                  <div className="mt-3 rounded-[var(--radius-md)] bg-[var(--color-background)] p-3">
                    <span className="text-xs font-medium uppercase tracking-wider text-[var(--color-text-muted)]">Lessons Learned</span>
                    <p className="mt-1 text-sm text-[var(--color-text-muted)]">{entry.lessons_learned}</p>
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
