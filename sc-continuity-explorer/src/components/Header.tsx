'use client';

import Link from 'next/link';
import { useState } from 'react';
import { Menu, X } from 'lucide-react';
import { ThemeToggle } from './ThemeToggle';

export function Header() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <header className="border-b border-[rgb(var(--border))] bg-[rgb(var(--background))] sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-6">
            <Link href="/explorer/" className="hover:opacity-80 transition-opacity">
              <h1 className="text-lg sm:text-xl font-semibold leading-tight">
                iAODE Continuity Explorer
              </h1>
              <p className="text-xs sm:text-sm text-[rgb(var(--muted-foreground))] mt-0.5">
                Explore trajectory structures across embedding methods
              </p>
            </Link>
            
            {/* Desktop Navigation */}
            <nav className="hidden md:flex items-center gap-4 ml-8">
              <a 
                href="/iAODE/" 
                className="text-sm font-medium text-[rgb(var(--muted-foreground))] hover:text-[rgb(var(--foreground))] transition-colors"
              >
                ← Main
              </a>
              <a 
                href="/iAODE/datasets/" 
                className="text-sm font-medium text-[rgb(var(--muted-foreground))] hover:text-[rgb(var(--foreground))] transition-colors"
              >
                Datasets
              </a>
            </nav>
          </div>
          
          <div className="flex items-center gap-2">
            <ThemeToggle />
            
            {/* Mobile Menu Button */}
            <button 
              className="md:hidden p-2 rounded-lg hover:bg-[rgb(var(--muted))] transition-colors" 
              aria-label="Toggle menu"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              type="button"
            >
              {mobileMenuOpen ? (
                <X className="w-6 h-6 text-[rgb(var(--foreground))]" />
              ) : (
                <Menu className="w-6 h-6 text-[rgb(var(--foreground))]" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile Navigation Menu */}
        {mobileMenuOpen && (
          <div className="md:hidden pt-4 pb-2 border-t border-[rgb(var(--border))] mt-4">
            <nav className="flex flex-col space-y-3">
              <a 
                href="/iAODE/" 
                onClick={() => setMobileMenuOpen(false)}
                className="text-base font-medium text-[rgb(var(--muted-foreground))] hover:text-[rgb(var(--foreground))] transition-colors px-2 py-1"
              >
                ← Back to Main
              </a>
              <a 
                href="/iAODE/datasets/" 
                onClick={() => setMobileMenuOpen(false)}
                className="text-base font-medium text-[rgb(var(--muted-foreground))] hover:text-[rgb(var(--foreground))] transition-colors px-2 py-1"
              >
                Dataset Browser
              </a>
            </nav>
          </div>
        )}
      </div>
    </header>
  );
}
