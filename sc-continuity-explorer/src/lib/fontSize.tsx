// ============================================================================
// lib/fontSize.tsx - Font Size Management
// ============================================================================

'use client';

import { createContext, useContext, useEffect, useState, ReactNode } from 'react';

type FontSize = 'small' | 'medium' | 'large' | 'x-large';

type FontSizeContextType = {
  fontSize: FontSize;
  setFontSize: (size: FontSize) => void;
  mounted: boolean;
};

// Extend Window interface for TypeScript
declare global {
  interface Window {
    __INITIAL_FONT_SIZE__?: FontSize;
  }
}

const FontSizeContext = createContext<FontSizeContextType | undefined>(undefined);

type FontSizeProviderProps = {
  children: ReactNode;
  defaultFontSize?: FontSize;
  storageKey?: string;
};

export function FontSizeProvider({ 
  children, 
  defaultFontSize = 'medium',
  storageKey = 'app-font-size'
}: FontSizeProviderProps) {
  // 🔥 KEY: Initialize with the font size set by blocking script
  const [fontSize, setFontSizeState] = useState<FontSize>(() => {
    // This runs on client, reads the value set by blocking script
    if (typeof window !== 'undefined' && window.__INITIAL_FONT_SIZE__) {
      return window.__INITIAL_FONT_SIZE__;
    }
    return defaultFontSize; // Fallback
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
      localStorage.setItem(storageKey, fontSize);
    } catch (error) {
      console.error('Error saving font size:', error);
    }
  }, [fontSize, mounted, storageKey]);

  // Listen for changes in other tabs/windows
  useEffect(() => {
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === storageKey && e.newValue) {
        setFontSizeState(e.newValue as FontSize);
      }
    };
    
    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, [storageKey]);

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