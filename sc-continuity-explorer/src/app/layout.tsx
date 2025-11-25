// ============================================================================
// FILE: app/layout.tsx
// Root layout with proper alignment and theme support
// ============================================================================

import type { Metadata } from 'next';
import { Header } from '@/components/Header';
import { ThemeProvider } from '@/lib/theme';
import './globals.css';

export const metadata: Metadata = {
  title: 'iAODE Continuity Explorer',
  description: 'Explore how continuity affects trajectory structure in single-cell data with iAODE',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen flex flex-col">
        <ThemeProvider defaultTheme="light" storageKey="sc-explorer-theme">
          <Header />

          {/* Main content */}
          <main className="flex-1 w-full">
            {children}
          </main>

          {/* Footer */}
          <footer className="border-t border-[rgb(var(--border))] bg-[rgb(var(--muted))] py-6">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-sm text-[rgb(var(--muted-foreground))]">
              Single-Cell Continuity Explorer © {new Date().getFullYear()}
            </div>
          </footer>
        </ThemeProvider>
      </body>
    </html>
  );
}