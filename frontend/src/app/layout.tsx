
import type { Metadata } from 'next';
import './globals.css';
import { ThemeProvider } from '@/components/theme-provider';

export const metadata: Metadata = {
  metadataBase: new URL('https://peterponyu.github.io'),
  title: {
    default: 'iAODE Workspace | Local-First Training',
    template: '%s | iAODE Workspace',
  },
  description: 'Local-first training workspace for iAODE model runs, with public-safe links back to the iAODE Pages surface and SCPortal.',
  alternates: {
    canonical: '/iAODE/frontend/',
  },
  openGraph: {
    title: 'iAODE Workspace | Local-First Training',
    description: 'Local-first training workspace for iAODE model runs, with public-safe links back to the iAODE Pages surface and SCPortal.',
    url: 'https://peterponyu.github.io/iAODE/frontend/',
    siteName: 'iAODE Workspace',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'iAODE Workspace | Local-First Training',
    description: 'Local-first training workspace for iAODE model runs, with public-safe links back to the iAODE Pages surface and SCPortal.',
  },
  robots: {
    index: false,
    follow: false,
  },
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
                  // Get stored theme or system preference
                  const storedTheme = localStorage.getItem('theme');
                  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
                  const theme = storedTheme || (prefersDark ? 'dark' : 'light');
                  
                  // Apply theme class immediately
                  if (theme === 'dark') {
                    document.documentElement.classList.add('dark');
                  } else {
                    document.documentElement.classList.remove('dark');
                  }
                } catch (e) {
                  console.error('Theme initialization error:', e);
                }
              })();
            `,
          }}
        />
      </head>
      <body className="antialiased">
        <ThemeProvider>
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
