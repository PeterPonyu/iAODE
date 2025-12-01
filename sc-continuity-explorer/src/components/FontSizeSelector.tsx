'use client';

// components/FontSizeSelector.tsx

import { useFontSize } from '@/lib/fontSize';
import { Type } from 'lucide-react';

const fontSizes = [
  { value: 'small', label: 'Small', display: 'A' },
  { value: 'medium', label: 'Medium', display: 'A' },
  { value: 'large', label: 'Large', display: 'A' },
  { value: 'x-large', label: 'Extra Large', display: 'A' },
] as const;

export function FontSizeSelector() {
  const { fontSize, setFontSize, mounted } = useFontSize();

  if (!mounted) return null;

  return (
    <div className="relative group">
      {/* Trigger Button */}
      <button
        className="p-2 rounded-lg hover:bg-[rgb(var(--muted))] transition-colors"
        aria-label="Change font size"
        title="Change text size"
        type="button"
      >
        <Type className="w-5 h-5 text-[rgb(var(--foreground))]" />
      </button>

      {/* Dropdown Menu */}
      <div className="absolute right-0 mt-2 w-40 bg-[rgb(var(--card))] border border-[rgb(var(--border))] rounded-lg shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-50">
        <div className="py-1">
          {fontSizes.map((size) => (
            <button
              key={size.value}
              onClick={() => setFontSize(size.value)}
              className={`
                w-full px-4 py-2 text-left transition-colors flex items-center justify-between
                ${fontSize === size.value
                  ? 'bg-[rgb(var(--primary))] text-white'
                  : 'text-[rgb(var(--foreground))] hover:bg-[rgb(var(--muted))]'
                }
              `}
              type="button"
            >
              <span className="text-sm font-medium">{size.label}</span>
              <span
                className="font-bold"
                style={{
                  fontSize: size.value === 'small' ? '0.875rem' : 
                           size.value === 'medium' ? '1rem' :
                           size.value === 'large' ? '1.125rem' : '1.25rem'
                }}
              >
                {size.display}
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}