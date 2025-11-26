// ============================================================================
// FILE: app/layout.tsx
// Root layout with proper alignment and theme support
// ============================================================================

import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import { Header } from '@/components/Header';
import { ThemeProvider } from '@/lib/theme';
import './globals.css';

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

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
        {/* Blocking script - runs BEFORE any React hydration */}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                try {
                  const storedTheme = localStorage.getItem('sc-explorer-theme');
                  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
                  const theme = storedTheme || (prefersDark ? 'dark' : 'light');
                  
                  if (theme === 'dark') {
                    document.documentElement.classList.add('dark');
                  } else {
                    document.documentElement.classList.remove('dark');
                  }
                  
                  window.__INITIAL_THEME__ = theme;
                } catch (e) {
                  console.error('Theme initialization error:', e);
                }
              })();
            `,
          }}
        />
      </head>
      <body 
        className={`${geistSans.variable} ${geistMono.variable} antialiased min-h-screen flex flex-col`}
        suppressHydrationWarning
      >
        <ThemeProvider defaultTheme="light" storageKey="sc-explorer-theme">
          <Header />

          {/* Main content */}
          <main className="flex-1 w-full">
            {children}
          </main>

          {/* Footer */}
          <footer className="border-t border-[rgb(var(--border))] bg-[rgb(var(--muted))] py-6">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-sm text-[rgb(var(--muted-foreground))]">
              iAODE Continuity Explorer © {new Date().getFullYear()}
            </div>
          </footer>
        </ThemeProvider>
      </body>
    </html>
  );
}