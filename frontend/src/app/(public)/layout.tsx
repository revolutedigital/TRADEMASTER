import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "../globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "TradeMaster - AI-Powered Crypto Trading",
  description:
    "AI-powered cryptocurrency trading platform with real-time analytics, risk management, and paper trading.",
};

export default function PublicLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className={inter.className}>
        <div className="min-h-screen bg-[var(--color-background)] text-[var(--color-text)]">
          {children}
        </div>
      </body>
    </html>
  );
}
