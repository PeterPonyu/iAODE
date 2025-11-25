'use client';

import { useSearchParams } from 'next/navigation';
import { getAllGSEGroups } from '@/lib/dataLoader';
import DatasetBrowser from './DatasetBrowser';
import { Suspense, useMemo } from 'react';

function DatasetBrowserContent() {
  const searchParams = useSearchParams();
  const dataType = (searchParams.get('type') === 'RNA' ? 'RNA' : 'ATAC') as 'ATAC' | 'RNA';
  
  const gseGroups = useMemo(() => getAllGSEGroups(dataType), [dataType]);
  const atacGroups = useMemo(() => getAllGSEGroups('ATAC'), []);
  const rnaGroups = useMemo(() => getAllGSEGroups('RNA'), []);
  
  const totalATACDatasets = useMemo(
    () => atacGroups.reduce((sum, g) => sum + g.datasets.length, 0),
    [atacGroups]
  );
  const totalRNADatasets = useMemo(
    () => rnaGroups.reduce((sum, g) => sum + g.datasets.length, 0),
    [rnaGroups]
  );

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="space-y-3">
        <h1 className="text-3xl font-bold text-[rgb(var(--foreground))] transition-colors">
          Browse Datasets
        </h1>
        <p className="text-[rgb(var(--muted-foreground))] transition-colors">
          Explore standardized single-cell datasets in 10X h5 format for iAODE-VAE benchmarking
        </p>
        
        {/* Quick Stats */}
        <div className="flex flex-wrap gap-6 text-sm">
          <div className="flex items-center gap-2">
            <span className="px-2 py-1 bg-[rgb(var(--atac-bg-subtle))] text-[rgb(var(--atac-text))] rounded border border-[rgb(var(--atac-border))] font-medium transition-colors">
              scATAC
            </span>
            <span className="text-[rgb(var(--text-secondary))] transition-colors">
              {atacGroups.length} studies · {totalATACDatasets} datasets
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="px-2 py-1 bg-[rgb(var(--rna-bg-subtle))] text-[rgb(var(--rna-text))] rounded border border-[rgb(var(--rna-border))] font-medium transition-colors">
              scRNA
            </span>
            <span className="text-[rgb(var(--text-secondary))] transition-colors">
              {rnaGroups.length} studies · {totalRNADatasets} datasets
            </span>
          </div>
        </div>
      </div>

      {/* Main Browser Component */}
      <DatasetBrowser 
        initialData={gseGroups} 
        dataType={dataType}
      />
    </div>
  );
}

export default function DatasetBrowserWrapper() {
  return (
    <Suspense fallback={<div className="p-8 text-center">Loading datasets...</div>}>
      <DatasetBrowserContent />
    </Suspense>
  );
}
