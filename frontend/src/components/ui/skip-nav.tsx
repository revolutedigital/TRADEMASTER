"use client";

/**
 * SkipNav - Accessible skip navigation link.
 *
 * Visually hidden by default, becomes visible when focused via keyboard Tab.
 * Allows screen-reader and keyboard users to jump directly to the main content
 * area, bypassing the sidebar and header navigation.
 */
export function SkipNav() {
  return (
    <a
      href="#main-content"
      className="sr-only focus:not-sr-only focus:absolute focus:z-[200] focus:p-4 focus:m-2 focus:rounded-lg focus:bg-[var(--color-surface)] focus:text-[var(--color-text)] focus:border focus:border-[var(--color-primary)] focus:shadow-lg focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
    >
      Skip to main content
    </a>
  );
}
