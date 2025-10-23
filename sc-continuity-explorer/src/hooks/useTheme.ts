/**
 * Theme management hook for light/dark mode
 * Fixed for SSR hydration
 */

'use client';

import { useEffect, useState } from 'react';

export type Theme = 'light' | 'dark' | 'system';

interface UseThemeReturn {
  theme: Theme;
  effectiveTheme: 'light' | 'dark';
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
  mounted: boolean; // ✅ NEW: Flag to check if client-side mounted
}

const STORAGE_KEY = 'sc-explorer-theme';

/**
 * Get the system's preferred color scheme
 */
function getSystemTheme(): 'light' | 'dark' {
  if (typeof window === 'undefined') return 'light';
  
  return window.matchMedia('(prefers-color-scheme: dark)').matches
    ? 'dark'
    : 'light';
}

/**
 * Get the effective theme (resolves 'system' to actual theme)
 */
function getEffectiveTheme(theme: Theme): 'light' | 'dark' {
  if (theme === 'system') {
    return getSystemTheme();
  }
  return theme;
}

/**
 * Hook for managing light/dark theme with SSR support
 * 
 * @example
 * function Header() {
 *   const { effectiveTheme, toggleTheme, mounted } = useTheme();
 *   
 *   // Don't render theme-specific content until mounted
 *   if (!mounted) return <div className="w-6 h-6" />;
 *   
 *   return (
 *     <button onClick={toggleTheme}>
 *       {effectiveTheme === 'dark' ? '🌙' : '☀️'}
 *     </button>
 *   );
 * }
 */
export function useTheme(): UseThemeReturn {
  // ✅ Track if component is mounted (client-side only)
  const [mounted, setMounted] = useState(false);

  // Initialize with safe default for SSR
  const [theme, setThemeState] = useState<Theme>('system');
  const [effectiveTheme, setEffectiveTheme] = useState<'light' | 'dark'>('light');

  // ✅ Set mounted flag after first render (client-side only)
  useEffect(() => {
    setMounted(true);

    // Load theme from localStorage (only runs on client)
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === 'light' || stored === 'dark' || stored === 'system') {
      setThemeState(stored);
    }
  }, []);

  // Update effective theme when theme changes
  useEffect(() => {
    if (!mounted) return;

    const newEffectiveTheme = getEffectiveTheme(theme);
    setEffectiveTheme(newEffectiveTheme);

    // Update document class
    const root = document.documentElement;
    root.classList.remove('light', 'dark');
    root.classList.add(newEffectiveTheme);

    // Update color-scheme for native elements
    root.style.colorScheme = newEffectiveTheme;
  }, [theme, mounted]);

  // Listen to system theme changes when theme is 'system'
  useEffect(() => {
    if (!mounted || theme !== 'system') return;

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    
    const handleChange = (e: MediaQueryListEvent) => {
      const newEffectiveTheme = e.matches ? 'dark' : 'light';
      setEffectiveTheme(newEffectiveTheme);
      
      const root = document.documentElement;
      root.classList.remove('light', 'dark');
      root.classList.add(newEffectiveTheme);
      root.style.colorScheme = newEffectiveTheme;
    };

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, [theme, mounted]);

  // Set theme and persist to localStorage
  const setTheme = (newTheme: Theme) => {
    setThemeState(newTheme);
    if (mounted) {
      localStorage.setItem(STORAGE_KEY, newTheme);
    }
  };

  // Toggle between light and dark (ignores system)
  const toggleTheme = () => {
    setTheme(effectiveTheme === 'dark' ? 'light' : 'dark');
  };

  return {
    theme,
    effectiveTheme,
    setTheme,
    toggleTheme,
    mounted, // ✅ Export mounted flag
  };
}