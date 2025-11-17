// components/datasets/DataTypeTabs.tsx
'use client';

import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { Activity, Dna } from 'lucide-react';

interface DataTypeTabsProps {
  currentType: 'ATAC' | 'RNA';
}

export default function DataTypeTabs({ currentType }: DataTypeTabsProps) {
  const searchParams = useSearchParams();
  
  // Preserve other search params when switching
  const createHref = (type: 'ATAC' | 'RNA') => {
    const params = new URLSearchParams(searchParams.toString());
    if (type === 'ATAC') {
      params.delete('type'); // ATAC is default
    } else {
      params.set('type', type);
    }
    const query = params.toString();
    return `/datasets${query ? `?${query}` : ''}`;
  };

  return (
    <div className="border-b border-[rgb(var(--border))] transition-colors">
      <nav className="flex gap-2" aria-label="Dataset type tabs">
        {/* ATAC Tab */}
        <Link
          href={createHref('ATAC')}
          className={`
            flex items-center gap-2 px-4 py-3 border-b-2 font-medium text-sm transition-all
            ${currentType === 'ATAC'
              ? 'border-[rgb(var(--atac-primary))] text-[rgb(var(--atac-primary))]'
              : 'border-transparent text-[rgb(var(--text-secondary))] hover:text-[rgb(var(--atac-text-bright))] hover:border-[rgb(var(--atac-border))]'
            }
          `}
        >
          <Activity className="w-4 h-4" />
          <span>scATAC-seq</span>
          <span className={`
            text-xs px-1.5 py-0.5 rounded
            ${currentType === 'ATAC'
              ? 'bg-[rgb(var(--atac-bg))] text-[rgb(var(--atac-text))]'
              : 'bg-[rgb(var(--muted))] text-[rgb(var(--text-tertiary))]'
            }
          `}>
            Peaks
          </span>
        </Link>

        {/* RNA Tab */}
        <Link
          href={createHref('RNA')}
          className={`
            flex items-center gap-2 px-4 py-3 border-b-2 font-medium text-sm transition-all
            ${currentType === 'RNA'
              ? 'border-[rgb(var(--rna-primary))] text-[rgb(var(--rna-primary))]'
              : 'border-transparent text-[rgb(var(--text-secondary))] hover:text-[rgb(var(--rna-text-bright))] hover:border-[rgb(var(--rna-border))]'
            }
          `}
        >
          <Dna className="w-4 h-4" />
          <span>scRNA-seq</span>
          <span className={`
            text-xs px-1.5 py-0.5 rounded
            ${currentType === 'RNA'
              ? 'bg-[rgb(var(--rna-bg))] text-[rgb(var(--rna-text))]'
              : 'bg-[rgb(var(--muted))] text-[rgb(var(--text-tertiary))]'
            }
          `}>
            Genes
          </span>
        </Link>
      </nav>
    </div>
  );
}