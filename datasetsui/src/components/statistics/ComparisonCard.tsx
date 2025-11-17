import { DatasetStats } from '@/types/datasets';
import { formatNumber, formatFileSize } from '@/lib/formatters';
import { ArrowRight, Activity, Dna } from 'lucide-react';

interface ComparisonCardProps {
  statsATAC: DatasetStats;
  statsRNA: DatasetStats;
}

export default function ComparisonCard({ statsATAC, statsRNA }: ComparisonCardProps) {
  const comparisons = [
    { 
      label: 'Studies', 
      atacValue: statsATAC.totalGSE, 
      rnaValue: statsRNA.totalGSE 
    },
    { 
      label: 'Datasets', 
      atacValue: statsATAC.totalDatasets, 
      rnaValue: statsRNA.totalDatasets 
    },
    { 
      label: 'Cells', 
      atacValue: statsATAC.totalCells, 
      rnaValue: statsRNA.totalCells, 
      format: true 
    },
    { 
      label: 'Features', 
      atacValue: statsATAC.totalFeatures, 
      rnaValue: statsRNA.totalFeatures, 
      format: true 
    },
    { 
      label: 'Data Size', 
      atacValue: statsATAC.totalSize, 
      rnaValue: statsRNA.totalSize, 
      formatSize: true 
    },
  ];

  return (
    <div className="card p-6 bg-gradient-to-r from-[rgb(var(--atac-bg-subtle))] via-[rgb(var(--card))] to-[rgb(var(--rna-bg-subtle))] border-2 border-[rgb(var(--border))] transition-colors">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
        <h2 className="text-xl font-bold text-[rgb(var(--foreground))] transition-colors">
          Data Type Comparison
        </h2>
        <div className="flex items-center gap-4 text-sm">
          <div className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-[rgb(var(--atac-primary))]" />
            <span className="font-medium text-[rgb(var(--foreground))] transition-colors">scATAC-seq</span>
          </div>
          <ArrowRight className="h-4 w-4 text-[rgb(var(--muted-foreground))]" />
          <div className="flex items-center gap-2">
            <Dna className="h-4 w-4 text-[rgb(var(--rna-primary))]" />
            <span className="font-medium text-[rgb(var(--foreground))] transition-colors">scRNA-seq</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        {comparisons.map((comp) => (
          <div key={comp.label} className="text-center">
            <p className="text-xs text-[rgb(var(--muted-foreground))] uppercase tracking-wide mb-2 transition-colors">
              {comp.label}
            </p>
            <div className="flex flex-col gap-1">
              <span className="text-lg font-bold text-[rgb(var(--atac-primary))] transition-colors">
                {comp.formatSize 
                  ? formatFileSize(comp.atacValue)
                  : comp.format 
                    ? formatNumber(comp.atacValue, true)
                    : comp.atacValue.toLocaleString()
                }
              </span>
              <span className="text-xs text-[rgb(var(--muted-foreground))] transition-colors">vs</span>
              <span className="text-lg font-bold text-[rgb(var(--rna-primary))] transition-colors">
                {comp.formatSize 
                  ? formatFileSize(comp.rnaValue)
                  : comp.format 
                    ? formatNumber(comp.rnaValue, true)
                    : comp.rnaValue.toLocaleString()
                }
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}