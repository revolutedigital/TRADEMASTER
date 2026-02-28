"use client";

import { useEffect, useState } from "react";
import { useAuthStore } from "@/stores/authStore";
import { LoginPage } from "@/components/login-page";

export function AuthGuard({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, restoreSession } = useAuthStore();
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    restoreSession();
    setChecked(true);
  }, [restoreSession]);

  if (!checked) {
    return (
      <div className="flex h-screen items-center justify-center bg-[var(--color-background)]">
        <div className="text-[var(--color-text-muted)]">Loading...</div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <LoginPage />;
  }

  return <>{children}</>;
}
