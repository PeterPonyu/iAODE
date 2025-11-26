
'use client';

import Link from 'next/link';
import { useState } from 'react';
import { Database, Menu, X } from 'lucide-react';
import { ThemeToggle } from './theme-toggle';

export function Header() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navLinks = [
    { href: '/', label: 'Home', external: false },
    { href: '/train', label: 'Training', external: false },
    { href: '/iAODE/datasets/', label: 'Datasets', external: true },
    { href: '/iAODE/explorer/', label: 'Explorer', external: true },
  ];

  return (
    <header className="sticky top-0 z-50 w-full border-b border-[rgb(var(--border))] bg-[rgb(var(--background))] shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          {/* Logo */}
          <Link 
            href="/" 
            className="flex items-center space-x-2 hover:opacity-80 transition-opacity"
          >
            <Database className="w-6 h-6 text-[rgb(var(--primary))]" />
            <span className="font-bold text-lg text-[rgb(var(--foreground))] whitespace-nowrap">
              iAODE Training
            </span>
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center space-x-6">
            {navLinks.map(link => (
              link.external ? (
                <a
                  key={link.href}
                  href={link.href}
                  className="text-sm font-medium transition-colors hover:text-[rgb(var(--primary-hover))] text-[rgb(var(--text-secondary))]"
                >
                  {link.label}
                </a>
              ) : (
                <Link
                  key={link.href}
                  href={link.href}
                  className="text-sm font-medium transition-colors hover:text-[rgb(var(--primary-hover))] text-[rgb(var(--text-secondary))]"
                >
                  {link.label}
                </Link>
              )
            ))}
          </nav>

          {/* Right side: Dark Mode Toggle + Mobile Menu */}
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
          <div className="md:hidden py-4 border-t border-[rgb(var(--border))]">
            <nav className="flex flex-col space-y-3">
              {navLinks.map(link => (
                link.external ? (
                  <a
                    key={link.href}
                    href={link.href}
                    onClick={() => setMobileMenuOpen(false)}
                    className="text-base font-medium transition-colors hover:text-[rgb(var(--primary-hover))] px-2 py-1 text-[rgb(var(--text-secondary))]"
                  >
                    {link.label}
                  </a>
                ) : (
                  <Link
                    key={link.href}
                    href={link.href}
                    onClick={() => setMobileMenuOpen(false)}
                    className="text-base font-medium transition-colors hover:text-[rgb(var(--primary-hover))] px-2 py-1 text-[rgb(var(--text-secondary))]"
                  >
                    {link.label}
                  </Link>
                )
              ))}
            </nav>
          </div>
        )}
      </div>
    </header>
  );
}
