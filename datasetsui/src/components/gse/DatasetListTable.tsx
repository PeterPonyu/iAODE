import { MergedDataset } from '@/types/datasets';
import { Badge } from '@/components/ui/Badge';
import { formatNumber, formatFileSize } from '@/lib/formatters';
import { Download, ChevronUp, ChevronDown } from 'lucide-react';
import { cn } from '@/lib/utils';

interface DatasetListTableProps {
  datasets: MergedDataset[];
  dataType: 'ATAC' | 'RNA';
  sortBy: 'cells' | 'features' | 'size' | 'filename';
  sortOrder: 'asc' | 'desc';
  onSort: (field: 'cells' | 'features' | 'size' | 'filename') => void;
}

export default function DatasetListTable({
  datasets,
  dataType,
  sortBy,
  sortOrder,
  onSort,
}: DatasetListTableProps) {
  const SortIcon = sortOrder === 'asc' ? ChevronUp : ChevronDown;
  const featureLabel = dataType === 'ATAC' ? 'Peaks' : 'Genes';
  const featureColorClass = dataType === 'ATAC'
    ? 'text-[rgb(var(--atac-primary))]'
    : 'text-[rgb(var(--rna-primary))]';

  const SortButton = ({ field, children }: { field: typeof sortBy; children: React.ReactNode }) => (
    <button
      onClick={() => onSort(field)}
      className="inline-flex items-center gap-1 hover:text-[rgb(var(--foreground))] transition-colors"
    >
      {children}
      {sortBy === field && <SortIcon className="h-4 w-4" />}
    </button>
  );

  return (
    <div className="overflow-x-auto -mx-6 px-6">
      <table className="w-full">
        <thead className="border-b border-[rgb(var(--border))] transition-colors">
          <tr>
            <th className="pb-3 text-left text-xs font-medium text-[rgb(var(--text-muted))] uppercase tracking-wider transition-colors">
              <SortButton field="filename">Filename</SortButton>
            </th>
            <th className="pb-3 text-left text-xs font-medium text-[rgb(var(--text-muted))] uppercase tracking-wider transition-colors">
              Category
            </th>
            <th className="pb-3 text-right text-xs font-medium text-[rgb(var(--text-muted))] uppercase tracking-wider transition-colors">
              <SortButton field="cells">Cells</SortButton>
            </th>
            <th className={cn(
              "pb-3 text-right text-xs font-medium uppercase tracking-wider transition-colors",
              featureColorClass
            )}>
              <SortButton field="features">{featureLabel}</SortButton>
            </th>
            <th className="pb-3 text-right text-xs font-medium text-[rgb(var(--text-muted))] uppercase tracking-wider transition-colors">
              <SortButton field="size">Size</SortButton>
            </th>
            <th className="pb-3 text-left text-xs font-medium text-[rgb(var(--text-muted))] uppercase tracking-wider transition-colors">
              Source
            </th>
            <th className="pb-3 text-right text-xs font-medium text-[rgb(var(--text-muted))] uppercase tracking-wider transition-colors">
              Actions
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-[rgb(var(--border))] transition-colors">
          {datasets.map((dataset) => (
            <tr key={dataset.id} className="hover:bg-[rgb(var(--muted))] transition-colors">
              <td className="py-4 pr-4">
                <div className="max-w-md">
                  <p className="text-sm font-medium text-[rgb(var(--foreground))] transition-colors truncate">
                    {dataset.dataFileName}
                  </p>
                  {dataset.gsmId && (
                    <p className="text-xs text-[rgb(var(--muted-foreground))] transition-colors">
                      {dataset.gsmId}
                    </p>
                  )}
                </div>
              </td>
              <td className="py-4 pr-4">
              <Badge variant={dataset.category as 'tiny' | 'small' | 'medium' | 'large'}>
                {dataset.category}
              </Badge>
              </td>
              <td className="py-4 pr-4 text-right text-sm text-[rgb(var(--foreground))] transition-colors">
                {formatNumber(dataset.nCells)}
              </td>
              <td className={cn(
                "py-4 pr-4 text-right text-sm font-medium transition-colors",
                featureColorClass
              )}>
                {formatNumber(dataset.nFeatures)}
              </td>
              <td className="py-4 pr-4 text-right text-sm text-[rgb(var(--foreground))] transition-colors">
                {formatFileSize(dataset.dataFileSize)}
              </td>
              <td className="py-4 pr-4">
                <p className="text-sm text-[rgb(var(--foreground))] transition-colors max-w-xs truncate">
                  {dataset.source !== 'Unknown' ? dataset.source : '-'}
                </p>
              </td>
              <td className="py-4 text-right">
                <a
                  href={dataset.downloadUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className={cn(
                    "inline-flex items-center gap-1.5 text-sm transition-colors",
                    dataType === 'ATAC'
                      ? 'text-[rgb(var(--atac-primary))] hover:text-[rgb(var(--atac-primary-hover))]'
                      : 'text-[rgb(var(--rna-primary))] hover:text-[rgb(var(--rna-primary-hover))]'
                  )}
                >
                  <Download className="h-4 w-4" />
                  <span className="hidden sm:inline">Download</span>
                </a>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}