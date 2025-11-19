'use client';

import { Search, X } from 'lucide-react';

interface DatasetSearchProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}

export default function DatasetSearch({ 
  value, 
  onChange, 
  placeholder = 'Search datasets...' 
}: DatasetSearchProps) {
  return (
    <div className="relative">
      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
        <Search className="h-5 w-5 text-[rgb(var(--text-muted))]" />
      </div>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="block w-full py-3 px-10 border border-[rgb(var(--border))] rounded-lg bg-[rgb(var(--background))] text-[rgb(var(--foreground))] text-sm outline-none transition-all placeholder:text-[rgb(var(--muted-foreground))] focus:outline-none focus:border-transparent focus:ring-2 focus:ring-[rgb(var(--primary))]"
      />
      {value && (
        <button
          onClick={() => onChange('')}
          className="absolute inset-y-0 right-0 pr-3 flex items-center text-[rgb(var(--text-muted))] transition-colors hover:text-[rgb(var(--text-tertiary))]"
          aria-label="Clear search"
        >
          <X className="h-5 w-5" />
        </button>
      )}
    </div>
  );
}