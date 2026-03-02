import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Sidebar } from "@/components/ui/sidebar";
import { Header } from "@/components/ui/header";
import { Providers } from "@/components/providers";
import { AuthGuard } from "@/components/auth-guard";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "TradeMaster - AI Trading Dashboard",
  description: "AI-powered cryptocurrency trading platform for BTC & ETH",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className={inter.className}>
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:absolute focus:z-50 focus:p-4 focus:bg-background focus:text-foreground focus:border"
        >
          Skip to main content
        </a>
        <AuthGuard>
          <Providers>
            <div className="flex h-screen overflow-hidden">
              <Sidebar />
              <div className="flex flex-1 flex-col overflow-hidden min-w-0">
                <Header />
                <main id="main-content" className="flex-1 overflow-y-auto p-4 sm:p-6" role="main">
                  {children}
                </main>
              </div>
            </div>
          </Providers>
        </AuthGuard>
      </body>
    </html>
  );
}
