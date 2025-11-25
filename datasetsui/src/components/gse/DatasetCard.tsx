import { MergedDataset } from '@/types/datasets';
import { Badge } from '@/components/ui/Badge';
import { formatNumber, formatFileSize } from '@/lib/formatters';
import { Download, FileText, Microscope, Activity, Dna, HardDrive, Cpu } from 'lucide-react';
import { cn } from '@/lib/utils';

interface DatasetCardProps {
  dataset: MergedDataset;
  dataType: 'ATAC' | 'RNA';
}

export default function DatasetCard({ dataset, dataType }: DatasetCardProps) {
  const {
    dataFileName,
    gsmId,
    nCells,
    nFeatures,
    dataFileSize,
    category,
    source,
    platform,
    downloadUrl,
  } = dataset;

  const featureLabel = dataType === 'ATAC' ? 'Peaks' : 'Genes';
  const FeatureIcon = dataType === 'ATAC' ? Activity : Dna;
  const featureColorClass = dataType === 'ATAC'
    ? 'text-[rgb(var(--atac-primary))]'
    : 'text-[rgb(var(--rna-primary))]';

  return (
    <div className="border border-[rgb(var(--border-light))] rounded-lg p-4 hover:border-[rgb(var(--card-hover-border))] transition-colors">
      {/* Header */}
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <FileText className="h-4 w-4 text-[rgb(var(--text-muted))] transition-colors flex-shrink-0" />
            <h3 className="text-sm font-medium text-[rgb(var(--foreground))] transition-colors truncate">
              {dataFileName}
            </h3>
          </div>
          {gsmId && (
            <p className="text-xs text-[rgb(var(--meta-label))] transition-colors">
              {gsmId}
            </p>
          )}
        </div>
        <Badge variant={category as 'tiny' | 'small' | 'medium' | 'large'}>
          {category}
        </Badge>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="flex items-center gap-2">
          <Microscope className="h-4 w-4 text-[rgb(var(--stat-green))] transition-colors" />
          <div>
            <p className="text-xs text-[rgb(var(--meta-label))] transition-colors">Cells</p>
            <p className="text-sm font-semibold text-[rgb(var(--foreground))] transition-colors">
              {formatNumber(nCells)}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <FeatureIcon className={cn("h-4 w-4 transition-colors", featureColorClass)} />
          <div>
            <p className="text-xs text-[rgb(var(--meta-label))] transition-colors">{featureLabel}</p>
            <p className={cn("text-sm font-semibold transition-colors", featureColorClass)}>
              {formatNumber(nFeatures)}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <HardDrive className="h-4 w-4 text-[rgb(var(--stat-orange))] transition-colors" />
          <div>
            <p className="text-xs text-[rgb(var(--meta-label))] transition-colors">Size</p>
            <p className="text-sm font-semibold text-[rgb(var(--foreground))] transition-colors">
              {formatFileSize(dataFileSize)}
            </p>
          </div>
        </div>
      </div>

      {/* Metadata */}
      {(source !== 'Unknown' || platform !== 'Unknown Platform') && (
        <div className="space-y-2 mb-4 pb-4 border-b border-[rgb(var(--border-light))] transition-colors">
          {source !== 'Unknown' && (
            <div className="flex items-start gap-2">
              <Dna className="h-4 w-4 text-[rgb(var(--text-muted))] transition-colors mt-0.5" />
              <div>
                <p className="text-xs text-[rgb(var(--meta-label))] transition-colors">Source</p>
                <p className="text-sm text-[rgb(var(--foreground))] transition-colors">{source}</p>
              </div>
            </div>
          )}
          {platform !== 'Unknown Platform' && (
            <div className="flex items-start gap-2">
              <Cpu className="h-4 w-4 text-[rgb(var(--text-muted))] transition-colors mt-0.5" />
              <div>
                <p className="text-xs text-[rgb(var(--meta-label))] transition-colors">Platform</p>
                <p className="text-sm text-[rgb(var(--foreground))] transition-colors">{platform}</p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Download Button */}
      <a
        href={downloadUrl}
        target="_blank"
        rel="noopener noreferrer"
        className={cn(
          "w-full inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg font-medium text-sm transition-colors",
          dataType === 'ATAC'
            ? 'bg-[rgb(var(--atac-primary))] hover:bg-[rgb(var(--atac-primary-hover))] text-white'
            : 'bg-[rgb(var(--rna-primary))] hover:bg-[rgb(var(--rna-primary-hover))] text-white'
        )}
      >
        <Download className="h-4 w-4" />
        Download from GEO
      </a>
    </div>
  );
}