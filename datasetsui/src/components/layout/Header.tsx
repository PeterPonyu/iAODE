'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Database, Moon, Sun, Menu } from 'lucide-react';
import { useState } from 'react';
import { useTheme } from '@/contexts/ThemeContext';

export default function Header() {
  const pathname = usePathname();
  const { theme, toggleTheme, mounted } = useTheme();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navLinks = [
    { href: '/', label: 'Home', external: false },
    { href: '/datasets', label: 'Datasets', external: false },
    { href: '/statistics', label: 'Statistics', external: false },
    { href: '/explorer/', label: 'Continuity Explorer', external: true }
  ];

  const isActive = (href: string, external: boolean) => {
    if (external) return false;
    if (href === '/') return pathname === '/';
    return pathname.startsWith(href);
  };

  const handleToggle = () => {
    console.log('Toggle button clicked');
    toggleTheme();
  };

  return (
    <header className="sticky top-0 z-50 w-full border-b border-[rgb(var(--border))] bg-[rgb(var(--background))] shadow-sm">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          {/* Logo */}
          <Link 
            href="/" 
            className="flex items-center space-x-2 hover:opacity-80 transition-opacity"
          >
            <Database className="w-6 h-6 text-[rgb(var(--primary))]" />
            <span className="font-bold text-lg text-[rgb(var(--foreground))] whitespace-nowrap">
              iAODE-VAE Benchmark Datasets
            </span>
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center space-x-6">
            {navLinks.map(link => (
              <Link
                key={link.href}
                href={link.href}
                className={`text-sm font-medium transition-colors hover:text-[rgb(var(--primary-hover))] ${
                  isActive(link.href, link.external)
                    ? 'text-[rgb(var(--primary))]'
                    : 'text-[rgb(var(--text-secondary))]'
                }`}
              >
                {link.label}
              </Link>
            ))}
          </nav>

          {/* Right side: Dark Mode Toggle + Mobile Menu */}
          <div className="flex items-center gap-2">
            {/* Dark Mode Toggle */}
            {mounted && (
              <button
                onClick={handleToggle}
                className="p-2 rounded-lg bg-[rgb(var(--secondary))] hover:bg-[rgb(var(--muted))] transition-colors"
                aria-label="Toggle dark mode"
                title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
                type="button"
              >
                {theme === 'light' ? (
                  <Moon className="w-5 h-5 text-[rgb(var(--foreground))]" />
                ) : (
                  <Sun className="w-5 h-5 text-[rgb(var(--theme-icon-sun))]" />
                )}
              </button>
            )}

            {/* Mobile Menu Button */}
            <button 
              className="md:hidden p-2 rounded-lg hover:bg-[rgb(var(--muted))] transition-colors" 
              aria-label="Open menu"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              type="button"
            >
              <Menu className="w-6 h-6 text-[rgb(var(--text-secondary))]" />
            </button>
          </div>
        </div>

        {/* Mobile Navigation Menu */}
        {mobileMenuOpen && (
          <div className="md:hidden py-4 border-t border-[rgb(var(--border))]">
            <nav className="flex flex-col space-y-3">
              {navLinks.map(link => (
                <Link
                  key={link.href}
                  href={link.href}
                  onClick={() => setMobileMenuOpen(false)}
                  className={`text-base font-medium transition-colors hover:text-[rgb(var(--primary-hover))] px-2 py-1 ${
                    isActive(link.href, link.external)
                      ? 'text-[rgb(var(--primary))]'
                      : 'text-[rgb(var(--text-secondary))]'
                  }`}
                >
                  {link.label}
                </Link>
              ))}
            </nav>
          </div>
        )}
      </div>
    </header>
  );
}