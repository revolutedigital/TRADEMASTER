"use client";

import { useState } from "react";

interface ExportButtonProps {
  endpoint: string;
  filename: string;
  label?: string;
  className?: string;
}

export function ExportButton({ endpoint, filename, label = "Exportar CSV", className = "" }: ExportButtonProps) {
  const [loading, setLoading] = useState(false);

  const handleExport = async () => {
    setLoading(true);
    try {
      const response = await fetch(endpoint, {
        credentials: "include",
      });
      if (!response.ok) throw new Error("Falha na exportação");

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      console.error("Export error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <button
      onClick={handleExport}
      disabled={loading}
      className={`inline-flex items-center gap-2 px-3 py-1.5 text-sm rounded-md border border-border
        hover:bg-accent transition-colors disabled:opacity-50 ${className}`}
      aria-label={label}
    >
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
      </svg>
      {loading ? "Exportando..." : label}
    </button>
  );
}
