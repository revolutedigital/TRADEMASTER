import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Sidebar } from "@/components/ui/sidebar";
import { Header } from "@/components/ui/header";
import { Providers } from "@/components/providers";
import { AuthGuard } from "@/components/auth-guard";
import { ToastProvider } from "@/components/ui/toast";
import { CommandPalette } from "@/components/ui/command-palette";
import { ErrorBoundary } from "@/components/ui/error-boundary";
import { OfflineBanner } from "@/components/ui/offline-banner";
import { ServiceWorkerRegister } from "@/components/sw-register";
import { SkipNav } from "@/components/ui/skip-nav";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const jetbrains = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "TradeMaster - AI Trading Dashboard",
  description: "AI-powered cryptocurrency trading platform for BTC & ETH",
  icons: {
    icon: "/favicon.svg",
    apple: "/logo-icon.svg",
  },
  openGraph: {
    title: "TradeMaster",
    description: "AI-powered cryptocurrency trading platform",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <head>
        <link rel="manifest" href="/manifest.json" />
        <meta name="theme-color" content="#0f172a" />
      </head>
      <body className={`${inter.variable} ${jetbrains.variable} ${inter.className}`}>
        <OfflineBanner />
        <ServiceWorkerRegister />
        <SkipNav />
        <AuthGuard>
          <ToastProvider>
            <Providers>
              <ErrorBoundary>
                <div className="flex h-screen overflow-hidden bg-[var(--color-background)]">
                  <Sidebar />
                  <div className="flex flex-1 flex-col overflow-hidden min-w-0">
                    <Header />
                    <main
                      id="main-content"
                      className="flex-1 overflow-y-auto p-4 sm:p-6 scroll-smooth"
                      role="main"
                    >
                      <div className="animate-fade-up">
                        {children}
                      </div>
                    </main>
                  </div>
                </div>
                <CommandPalette />
              </ErrorBoundary>
            </Providers>
          </ToastProvider>
        </AuthGuard>
      </body>
    </html>
  );
}
