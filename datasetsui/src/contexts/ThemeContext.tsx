'use client';

// contexts/ThemeContext.tsx

import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';

type Theme = 'light' | 'dark';

interface ThemeContextType {
  theme: Theme;
  toggleTheme: () => void;
  mounted: boolean;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>('light');
  const [mounted, setMounted] = useState(false);

  // Initialize theme on mount - runs only once
  useEffect(() => {
    const storedTheme = localStorage.getItem('theme') as Theme | null;
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const initialTheme = storedTheme || (prefersDark ? 'dark' : 'light');
    
    console.log('Initializing theme:', initialTheme);
    setTheme(initialTheme);
    setMounted(true);
  }, []); // ✅ Empty dependency array - runs once

  // Apply theme whenever it changes
  useEffect(() => {
    console.log('Applying theme to DOM:', theme);
    const root = document.documentElement;
    
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
  }, [theme]); // ✅ Only runs when theme changes

  // Toggle theme function
  const toggleTheme = () => {
    setTheme(prevTheme => {
      const newTheme = prevTheme === 'light' ? 'dark' : 'light';
      console.log(`Toggling: ${prevTheme} → ${newTheme}`);
      
      // Save to localStorage
      try {
        localStorage.setItem('theme', newTheme);
      } catch (error) {
        console.error('Error saving theme:', error);
      }
      
      return newTheme;
    });
  };

  const value = {
    theme,
    toggleTheme,
    mounted
  };

  // ✅ ALWAYS render with Provider wrapper
  return (
    <ThemeContext.Provider value={value}>
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