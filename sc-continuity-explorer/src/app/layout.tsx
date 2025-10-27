
// ============================================================================
// FILE: app/layout.tsx
// Root layout with proper alignment
// ============================================================================

import type { Metadata } from 'next';
import Link from 'next/link';
import { ThemeToggle } from '@/components/ThemeToggle';
import './globals.css';

export const metadata: Metadata = {
  title: 'Single-Cell Continuity Explorer',
  description: 'Explore how continuity affects trajectory structure in single-cell data',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen flex flex-col">
        {/* Header */}
        <header className="border-b border-[var(--color-border)] bg-[var(--color-background)] sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between gap-4">
            <Link href="/" className="hover:opacity-80 transition-opacity">
              <h1 className="text-lg sm:text-xl font-semibold leading-tight">
                Single-Cell Continuity Explorer
              </h1>
              <p className="text-xs sm:text-sm text-[var(--color-muted-foreground)] mt-0.5">
                Explore trajectory structures across embedding methods
              </p>
            </Link>
            <ThemeToggle />
          </div>
        </header>

        {/* Main content */}
        <main className="flex-1 w-full">
          {children}
        </main>

        {/* Footer */}
        <footer className="border-t border-[var(--color-border)] bg-[var(--color-muted)] py-6">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-sm text-[var(--color-muted-foreground)]">
            Single-Cell Continuity Explorer © {new Date().getFullYear()}
          </div>
        </footer>
      </body>
    </html>
  );
}
