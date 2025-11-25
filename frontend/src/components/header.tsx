
'use client';

import Link from 'next/link';
import { Home } from 'lucide-react';
import { ThemeToggle } from './theme-toggle';

export function Header() {
  return (
    <header style={{ borderBottomWidth: '1px', borderColor: 'var(--color-border)' }}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
        <Link href="/" className="flex items-center gap-2 text-2xl font-bold hover:opacity-80 transition-opacity">
          <Home className="w-6 h-6" />
          <span>iAODE</span>
        </Link>
        <ThemeToggle />
      </div>
    </header>
  );
}
