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
      className="inline-flex items-center gap-1 hover:text-gray-900 dark:hover:text-gray-100 transition-colors"
    >
      {children}
      {sortBy === field && <SortIcon className="h-4 w-4" />}
    </button>
  );

  return (
    <div className="overflow-x-auto -mx-6 px-6">
      <table className="w-full">
        <thead className="border-b border-gray-200 dark:border-gray-700">
          <tr>
            <th className="pb-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
              <SortButton field="filename">Filename</SortButton>
            </th>
            <th className="pb-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
              Category
            </th>
            <th className="pb-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
              <SortButton field="cells">Cells</SortButton>
            </th>
            <th className="pb-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
              <SortButton field="peaks">Peaks</SortButton>
            </th>
            <th className="pb-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
              <SortButton field="size">Size</SortButton>
            </th>
            <th className="pb-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
              Source
            </th>
            <th className="pb-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
              Actions
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
          {datasets.map((dataset) => (
            <tr key={dataset.id} className="hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors">
              <td className="py-4 pr-4">
                <div className="max-w-md">
                  <p className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
                    {dataset.dataFileName}
                  </p>
                  {dataset.gsmId && (
                    <p className="text-xs text-gray-500 dark:text-gray-400">
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
              <td className="py-4 pr-4 text-right text-sm text-gray-900 dark:text-gray-100">
                {formatNumber(dataset.nCells)}
              </td>
              <td className="py-4 pr-4 text-right text-sm text-gray-900 dark:text-gray-100">
                {formatNumber(dataset.nPeaks)}
              </td>
              <td className="py-4 pr-4 text-right text-sm text-gray-900 dark:text-gray-100">
                {formatFileSize(dataset.dataFileSize)}
              </td>
              <td className="py-4 pr-4">
                <p className="text-sm text-gray-900 dark:text-gray-100 max-w-xs truncate">
                  {dataset.source !== 'Unknown' ? dataset.source : '-'}
                </p>
              </td>
              <td className="py-4 text-right">
                <a
                  href={dataset.downloadUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 text-sm text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300"
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