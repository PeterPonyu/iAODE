'use client';

// contexts/FontSizeContext.tsx

import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';

type FontSize = 'small' | 'medium' | 'large' | 'x-large';

interface FontSizeContextType {
  fontSize: FontSize;
  setFontSize: (size: FontSize) => void;
  mounted: boolean;
}

// Extend Window interface for TypeScript
declare global {
  interface Window {
    __INITIAL_FONT_SIZE__?: FontSize;
  }
}

const FontSizeContext = createContext<FontSizeContextType | undefined>(undefined);

export function FontSizeProvider({ children }: { children: ReactNode }) {
  // 🔥 KEY: Initialize with the font size set by blocking script
  const [fontSize, setFontSizeState] = useState<FontSize>(() => {
    // This runs on client, reads the value set by blocking script
    if (typeof window !== 'undefined' && window.__INITIAL_FONT_SIZE__) {
      return window.__INITIAL_FONT_SIZE__;
    }
    return 'medium'; // Fallback
  });
  
  const [mounted, setMounted] = useState(false);

  // Mark as mounted
  useEffect(() => {
    setMounted(true);
  }, []);

  // Apply font size changes (after initial load)
  useEffect(() => {
    if (!mounted) return; // Skip on first render (already set by blocking script)
    
    const root = document.documentElement;
    root.setAttribute('data-font-size', fontSize);
    
    // Save to localStorage
    try {
      localStorage.setItem('font-size', fontSize);
    } catch (error) {
      console.error('Error saving font size:', error);
    }
  }, [fontSize, mounted]);

  // Listen for changes in other tabs/windows
  useEffect(() => {
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'font-size' && e.newValue) {
        setFontSizeState(e.newValue as FontSize);
      }
    };
    
    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);

  // Set font size function
  const setFontSize = (size: FontSize) => {
    setFontSizeState(size);
  };

  const value = {
    fontSize,
    setFontSize,
    mounted
  };

  return (
    <FontSizeContext.Provider value={value}>
      {children}
    </FontSizeContext.Provider>
  );
}

export function useFontSize() {
  const context = useContext(FontSizeContext);
  if (context === undefined) {
    throw new Error('useFontSize must be used within a FontSizeProvider');
  }
  return context;
}