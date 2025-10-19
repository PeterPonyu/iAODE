import { MergedDataset } from '@/types/datasets';
import { Badge } from '@/components/ui/Badge';
import { formatNumber, formatFileSize } from '@/lib/formatters';
import { Download, FileText, Microscope, BarChart3, HardDrive, Dna, Cpu } from 'lucide-react';

interface DatasetCardProps {
  dataset: MergedDataset;
}

export default function DatasetCard({ dataset }: DatasetCardProps) {
  const {
    dataFileName,
    gsmId,
    nCells,
    nPeaks,
    dataFileSize,
    category,
    source,
    platform,
    downloadUrl,
  } = dataset;

  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:border-blue-300 dark:hover:border-blue-600 transition-colors">
      {/* Header */}
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <FileText className="h-4 w-4 text-gray-400 flex-shrink-0" />
            <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
              {dataFileName}
            </h3>
          </div>
          {gsmId && (
            <p className="text-xs text-gray-500 dark:text-gray-400">
              {gsmId}
            </p>
          )}
        </div>
        <Badge variant={category as any}>
          {category}
        </Badge>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="flex items-center gap-2">
          <Microscope className="h-4 w-4 text-gray-400" />
          <div>
            <p className="text-xs text-gray-500 dark:text-gray-400">Cells</p>
            <p className="text-sm font-semibold text-gray-900 dark:text-gray-100">
              {formatNumber(nCells)}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <BarChart3 className="h-4 w-4 text-gray-400" />
          <div>
            <p className="text-xs text-gray-500 dark:text-gray-400">Peaks</p>
            <p className="text-sm font-semibold text-gray-900 dark:text-gray-100">
              {formatNumber(nPeaks)}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <HardDrive className="h-4 w-4 text-gray-400" />
          <div>
            <p className="text-xs text-gray-500 dark:text-gray-400">Size</p>
            <p className="text-sm font-semibold text-gray-900 dark:text-gray-100">
              {formatFileSize(dataFileSize)}
            </p>
          </div>
        </div>
      </div>

      {/* Metadata */}
      {(source !== 'Unknown' || platform !== 'Unknown Platform') && (
        <div className="space-y-2 mb-4 pb-4 border-b border-gray-200 dark:border-gray-700">
          {source !== 'Unknown' && (
            <div className="flex items-start gap-2">
              <Dna className="h-4 w-4 text-gray-400 mt-0.5" />
              <div>
                <p className="text-xs text-gray-500 dark:text-gray-400">Source</p>
                <p className="text-sm text-gray-900 dark:text-gray-100">{source}</p>
              </div>
            </div>
          )}
          {platform !== 'Unknown Platform' && (
            <div className="flex items-start gap-2">
              <Cpu className="h-4 w-4 text-gray-400 mt-0.5" />
              <div>
                <p className="text-xs text-gray-500 dark:text-gray-400">Platform</p>
                <p className="text-sm text-gray-900 dark:text-gray-100">{platform}</p>
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
        className="btn-primary w-full inline-flex items-center justify-center gap-2"
      >
        <Download className="h-4 w-4" />
        Download from GEO
      </a>
    </div>
  );
}