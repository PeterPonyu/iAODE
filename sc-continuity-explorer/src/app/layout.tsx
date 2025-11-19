// ============================================================================
// FILE: app/layout.tsx
// Root layout with theme support
// ============================================================================

import type { Metadata } from 'next';
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
      <body>
        <div className="min-h-screen flex flex-col">
          {/* Header */}
          <header className="border-b border-[var(--color-border)] bg-[var(--color-background)]">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
              <div>
                <h1 className="text-xl font-semibold">Single-Cell Continuity Explorer</h1>
                <p className="text-sm text-[var(--color-muted-foreground)]">
                  Explore trajectory structures across embedding methods
                </p>
              </div>
              <ThemeToggle />
            </div>
          </header>

          {/* Main content */}
          <main className="flex-1">
            {children}
          </main>

          {/* Footer */}
          <footer className="border-t border-[var(--color-border)] bg-[var(--color-muted)] py-6">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-sm text-[var(--color-muted-foreground)]">
              Single-Cell Continuity Explorer © {new Date().getFullYear()}
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}
