// app/datasets/page.tsx
import { getAllGSEGroups } from '@/lib/dataLoader';
import DatasetBrowser from '@/components/datasets/DatasetBrowser';

export const metadata = {
  title: 'Browse Datasets | iAODE-VAE Benchmark',
  description: 'Browse single-cell ATAC-seq and RNA-seq datasets from NCBI GEO',
};

interface DatasetsPageProps {
  searchParams: { type?: 'ATAC' | 'RNA' };
}

export default function DatasetsPage({ searchParams }: DatasetsPageProps) {
  // Default to ATAC if not specified
  const dataType = searchParams.type === 'RNA' ? 'RNA' : 'ATAC';
  
  const gseGroups = getAllGSEGroups(dataType);
  const totalDatasets = gseGroups.reduce((sum, g) => sum + g.datasets.length, 0);
  
  // Get stats for both types for the header
  const atacGroups = getAllGSEGroups('ATAC');
  const rnaGroups = getAllGSEGroups('RNA');
  const totalATACDatasets = atacGroups.reduce((sum, g) => sum + g.datasets.length, 0);
  const totalRNADatasets = rnaGroups.reduce((sum, g) => sum + g.datasets.length, 0);
  
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
        totalDatasets={totalDatasets}
      />
    </div>
  );
}