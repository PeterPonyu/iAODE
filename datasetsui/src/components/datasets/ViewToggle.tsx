'use client';

import { Grid3x3, List } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ViewToggleProps {
  mode: 'grid' | 'table';
  onChange: (mode: 'grid' | 'table') => void;
}

export default function ViewToggle({ mode, onChange }: ViewToggleProps) {
  return (
    <div className="inline-flex rounded-lg border border-[rgb(var(--border))] bg-[rgb(var(--card))] p-1 transition-all">
      <button
        onClick={() => onChange('grid')}
        className={cn(
          'inline-flex items-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium transition-all cursor-pointer border-none outline-none',
          'focus-visible:outline-2 focus-visible:outline-[rgb(var(--primary))] focus-visible:outline-offset-2',
          mode === 'grid'
            ? 'bg-[rgb(var(--primary))] text-[rgb(var(--primary-foreground))] shadow-sm'
            : 'bg-transparent text-[rgb(var(--text-secondary))] hover:bg-[rgb(var(--muted))]'
        )}
        aria-label="Grid view"
      >
        <Grid3x3 className="h-4 w-4" />
        <span className="hidden sm:inline">Grid</span>
      </button>
      <button
        onClick={() => onChange('table')}
        className={cn(
          'inline-flex items-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium transition-all cursor-pointer border-none outline-none',
          'focus-visible:outline-2 focus-visible:outline-[rgb(var(--primary))] focus-visible:outline-offset-2',
          mode === 'table'
            ? 'bg-[rgb(var(--primary))] text-[rgb(var(--primary-foreground))] shadow-sm'
            : 'bg-transparent text-[rgb(var(--text-secondary))] hover:bg-[rgb(var(--muted))]'
        )}
        aria-label="Table view"
      >
        <List className="h-4 w-4" />
        <span className="hidden sm:inline">Table</span>
      </button>
    </div>
  );
}
