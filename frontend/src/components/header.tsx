
'use client';

import Link from 'next/link';
import { Sparkles } from 'lucide-react';
import { ThemeToggle } from './theme-toggle';

export function Header() {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-[rgb(var(--border))] bg-[rgb(var(--background))] shadow-sm">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          {/* Logo */}
          <Link 
            href="/" 
            className="flex items-center space-x-2 hover:opacity-80 transition-opacity"
          >
            <Sparkles className="w-6 h-6 text-[rgb(var(--training-primary))]" />
            <span className="font-bold text-lg text-[rgb(var(--foreground))] whitespace-nowrap">
              iAODE Training
            </span>
          </Link>

          {/* Navigation Links */}
          <nav className="hidden md:flex items-center space-x-6">
            <Link
              href="/"
              className="text-sm font-medium transition-colors hover:text-[rgb(var(--primary-hover))] text-[rgb(var(--text-secondary))]"
            >
              Home
            </Link>
            <Link
              href="/train"
              className="text-sm font-medium transition-colors hover:text-[rgb(var(--primary-hover))] text-[rgb(var(--text-secondary))]"
            >
              Train Model
            </Link>
          </nav>

          {/* Theme Toggle */}
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
