import { MergedDataset } from '@/types/datasets';
import { Badge } from '@/components/ui/Badge';
import { formatNumber, formatFileSize } from '@/lib/formatters';
import { Download, ChevronUp, ChevronDown } from 'lucide-react';
import { cn } from '@/lib/utils';

interface DatasetListTableProps {
  datasets: MergedDataset[];
  sortBy: 'cells' | 'peaks' | 'size' | 'filename';
  sortOrder: 'asc' | 'desc';
  onSort: (field: 'cells' | 'peaks' | 'size' | 'filename') => void;
}

export default function DatasetListTable({
  datasets,
  sortBy,
  sortOrder,
  onSort,
}: DatasetListTableProps) {
  const SortIcon = sortOrder === 'asc' ? ChevronUp : ChevronDown;

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
        <thead className="border-b border-[rgb(var(--border))]">
          <tr>
            <th className="pb-3 text-left text-xs font-medium text-[rgb(var(--text-muted))] uppercase tracking-wider">
              <SortButton field="filename">Filename</SortButton>
            </th>
            <th className="pb-3 text-left text-xs font-medium text-[rgb(var(--text-muted))] uppercase tracking-wider">
              Category
            </th>
            <th className="pb-3 text-right text-xs font-medium text-[rgb(var(--text-muted))] uppercase tracking-wider">
              <SortButton field="cells">Cells</SortButton>
            </th>
            <th className="pb-3 text-right text-xs font-medium text-[rgb(var(--text-muted))] uppercase tracking-wider">
              <SortButton field="peaks">Peaks</SortButton>
            </th>
            <th className="pb-3 text-right text-xs font-medium text-[rgb(var(--text-muted))] uppercase tracking-wider">
              <SortButton field="size">Size</SortButton>
            </th>
            <th className="pb-3 text-left text-xs font-medium text-[rgb(var(--text-muted))] uppercase tracking-wider">
              Source
            </th>
            <th className="pb-3 text-right text-xs font-medium text-[rgb(var(--text-muted))] uppercase tracking-wider">
              Actions
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-[rgb(var(--border))]">
          {datasets.map((dataset) => (
            <tr key={dataset.id} className="hover:bg-[rgb(var(--muted))] transition-colors">
              <td className="py-4 pr-4">
                <div className="max-w-md">
                  <p className="text-sm font-medium text-[rgb(var(--foreground))] truncate">
                    {dataset.dataFileName}
                  </p>
                  {dataset.gsmId && (
                    <p className="text-xs text-[rgb(var(--muted-foreground))]">
                      {dataset.gsmId}
                    </p>
                  )}
                </div>
              </td>
              <td className="py-4 pr-4">
                <Badge variant={dataset.category as any}>
                  {dataset.category}
                </Badge>
              </td>
              <td className="py-4 pr-4 text-right text-sm text-[rgb(var(--foreground))]">
                {formatNumber(dataset.nCells)}
              </td>
              <td className="py-4 pr-4 text-right text-sm text-[rgb(var(--foreground))]">
                {formatNumber(dataset.nPeaks)}
              </td>
              <td className="py-4 pr-4 text-right text-sm text-[rgb(var(--foreground))]">
                {formatFileSize(dataset.dataFileSize)}
              </td>
              <td className="py-4 pr-4">
                <p className="text-sm text-[rgb(var(--foreground))] max-w-xs truncate">
                  {dataset.source !== 'Unknown' ? dataset.source : '-'}
                </p>
              </td>
              <td className="py-4 text-right">
                <a
                  href={dataset.downloadUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 text-sm text-[rgb(var(--primary))] hover:text-[rgb(var(--primary-hover))] transition-colors"
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