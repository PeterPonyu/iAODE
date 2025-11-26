// ============================================================================
// lib/theme.tsx - Theme Management
// ============================================================================

'use client';

import { createContext, useContext, useEffect, useState, ReactNode } from 'react';

type Theme = 'light' | 'dark';

// Extend Window interface for TypeScript
declare global {
  interface Window {
    __INITIAL_THEME__?: Theme;
  }
}

type ThemeContextType = {
  theme: Theme;
  toggleTheme: () => void;
  setTheme: (theme: Theme) => void;
  mounted: boolean;
};

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

type ThemeProviderProps = {
  children: ReactNode;
  defaultTheme?: Theme;
  storageKey?: string;
};

export function ThemeProvider({ 
  children, 
  defaultTheme = 'light',
  storageKey = 'app-theme'
}: ThemeProviderProps) {
  // Initialize with the theme set by blocking script
  const [theme, setThemeState] = useState<Theme>(() => {
    if (typeof window !== 'undefined' && window.__INITIAL_THEME__) {
      return window.__INITIAL_THEME__;
    }
    return defaultTheme;
  });
  const [mounted, setMounted] = useState(false);

  // Mark as mounted
  useEffect(() => {
    setMounted(true);
  }, []);

  // Apply theme changes (after initial load)
  useEffect(() => {
    if (!mounted) return;
    
    const root = document.documentElement;
    
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    
    // Save to localStorage
    try {
      localStorage.setItem(storageKey, theme);
    } catch (error) {
      console.error('Error saving theme:', error);
    }
  }, [theme, mounted, storageKey]);

  // Listen for system preference changes
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    
    const handleChange = (e: MediaQueryListEvent) => {
      const hasStoredTheme = localStorage.getItem(storageKey);
      if (!hasStoredTheme) {
        setThemeState(e.matches ? 'dark' : 'light');
      }
    };
    
    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, [storageKey]);

  // Listen for changes in other tabs/windows
  useEffect(() => {
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === storageKey && e.newValue) {
        setThemeState(e.newValue as Theme);
      }
    };
    
    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, [storageKey]);

  const toggleTheme = () => {
    setThemeState(prev => prev === 'light' ? 'dark' : 'light');
  };

  const setTheme = (newTheme: Theme) => {
    setThemeState(newTheme);
  };

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme, setTheme, mounted }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  
  return context;
}