
'use client';

import { Moon, Sun } from 'lucide-react';
import { useTheme } from './theme-provider';

export function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();

  const toggleTheme = () => {
    setTheme(resolvedTheme === 'light' ? 'dark' : 'light');
  };

  return (
    <button
      onClick={toggleTheme}
      className="p-2 rounded-lg bg-[rgb(var(--secondary))] hover:bg-[rgb(var(--muted))] transition-colors"
      aria-label="Toggle dark mode"
      title={`Switch to ${resolvedTheme === 'light' ? 'dark' : 'light'} mode`}
      type="button"
    >
      {resolvedTheme === 'light' ? (
        <Moon className="w-5 h-5 text-[rgb(var(--foreground))]" />
      ) : (
        <Sun className="w-5 h-5 text-[rgb(var(--theme-icon-sun))]" />
      )}
    </button>
  );
}
