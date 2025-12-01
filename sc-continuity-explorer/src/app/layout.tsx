// ============================================================================
// FILE: app/layout.tsx
// Root layout with proper alignment and theme support
// ============================================================================

import type { Metadata } from 'next';
import { Header } from '@/components/Header';
import { ThemeProvider } from '@/lib/theme';
import { FontSizeProvider } from '@/lib/fontSize';
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
      <head>
        {/* 🔥 Blocking script - runs BEFORE any React hydration */}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                try {
                  // Theme initialization
                  const storedTheme = localStorage.getItem('sc-explorer-theme');
                  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
                  const theme = storedTheme || (prefersDark ? 'dark' : 'light');
                  
                  // Apply theme class immediately
                  if (theme === 'dark') {
                    document.documentElement.classList.add('dark');
                  } else {
                    document.documentElement.classList.remove('dark');
                  }
                  
                  // Store for ThemeContext to read
                  window.__INITIAL_THEME__ = theme;

                  // Font size initialization
                  const storedFontSize = localStorage.getItem('sc-explorer-font-size');
                  const fontSize = storedFontSize || 'medium';
                  
                  // Apply font size attribute immediately
                  document.documentElement.setAttribute('data-font-size', fontSize);
                  
                  // Store for FontSizeContext to read
                  window.__INITIAL_FONT_SIZE__ = fontSize;
                } catch (e) {
                  console.error('Initialization error:', e);
                }
              })();
            `,
          }}
        />
      </head>
      <body className="min-h-screen flex flex-col" suppressHydrationWarning>
        <ThemeProvider defaultTheme="light" storageKey="sc-explorer-theme">
          <FontSizeProvider storageKey="sc-explorer-font-size">
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
          </FontSizeProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}