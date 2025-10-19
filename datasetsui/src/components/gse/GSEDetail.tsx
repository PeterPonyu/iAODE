'use client';

import { useState, useMemo } from 'react';
import Link from 'next/link';
import { GSEGroup } from '@/types/datasets';
import { generateGeoUrl } from '@/lib/geoUtils';
import { formatNumber, formatFileSize } from '@/lib/formatters';
import GSEHeader from './GSEHeader';
import GSEStats from './GSEStats';
import DatasetList from './DatasetList';
import DatasetListTable from './DatasetListTable';
import { ArrowLeft, LayoutGrid, List } from 'lucide-react';
import { cn } from '@/lib/utils';

interface GSEDetailProps {
  gseGroup: GSEGroup;
}

export default function GSEDetail({ gseGroup }: GSEDetailProps) {
  const [viewMode, setViewMode] = useState<'card' | 'table'>('card');
  const [sortBy, setSortBy] = useState<'cells' | 'peaks' | 'size' | 'filename'>('cells');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  // Sort datasets
  const sortedDatasets = useMemo(() => {
    const sorted = [...gseGroup.datasets].sort((a, b) => {
      let aVal: number, bVal: number;

      switch (sortBy) {
        case 'cells':
          aVal = parseInt(a.nCells) || 0;
          bVal = parseInt(b.nCells) || 0;
          break;
        case 'peaks':
          aVal = parseInt(a.nPeaks) || 0;
          bVal = parseInt(b.nPeaks) || 0;
          break;
        case 'size':
          aVal = parseFloat(a.dataFileSize) || 0;
          bVal = parseFloat(b.dataFileSize) || 0;
          break;
        case 'filename':
          return sortOrder === 'asc'
            ? a.dataFileName.localeCompare(b.dataFileName)
            : b.dataFileName.localeCompare(a.dataFileName);
        default:
          aVal = 0;
          bVal = 0;
      }

      return sortOrder === 'asc' ? aVal - bVal : bVal - aVal;
    });

    return sorted;
  }, [gseGroup.datasets, sortBy, sortOrder]);

  const handleSort = (field: typeof sortBy) => {
    if (sortBy === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field);
      setSortOrder('desc');
    }
  };

  return (
    <div className="space-y-6">
      {/* Back Button */}
      <Link
        href="/datasets"
        className="inline-flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 transition-colors"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to Browse
      </Link>

      {/* Header Section */}
      <GSEHeader gseGroup={gseGroup} />

      {/* Stats Cards */}
      <GSEStats gseGroup={gseGroup} />

      {/* Datasets Section */}
      <div className="card p-6">
        {/* Section Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
          <div>
            <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-1">
              Datasets in this Study
            </h2>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              {gseGroup.datasets.length} dataset{gseGroup.datasets.length !== 1 ? 's' : ''} available
            </p>
          </div>

          {/* Controls */}
          <div className="flex items-center gap-3">
            {/* Sort Dropdown */}
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
              className="
                px-3 py-2 text-sm
                border border-gray-300 dark:border-gray-600
                rounded-lg
                bg-white dark:bg-gray-800
                text-gray-900 dark:text-gray-100
                focus:outline-none focus:ring-2 focus:ring-blue-500
              "
            >
              <option value="cells">Sort by Cells</option>
              <option value="peaks">Sort by Peaks</option>
              <option value="size">Sort by Size</option>
              <option value="filename">Sort by Filename</option>
            </select>

            {/* View Toggle */}
            <div className="inline-flex rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-1">
              <button
                onClick={() => setViewMode('card')}
                className={cn(
                  'inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
                  viewMode === 'card'
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                )}
                aria-label="Card view"
              >
                <LayoutGrid className="h-4 w-4" />
                <span className="hidden sm:inline">Cards</span>
              </button>
              <button
                onClick={() => setViewMode('table')}
                className={cn(
                  'inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
                  viewMode === 'table'
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                )}
                aria-label="Table view"
              >
                <List className="h-4 w-4" />
                <span className="hidden sm:inline">Table</span>
              </button>
            </div>
          </div>
        </div>

        {/* Dataset List */}
        {viewMode === 'card' ? (
          <DatasetList datasets={sortedDatasets} />
        ) : (
          <DatasetListTable 
            datasets={sortedDatasets} 
            sortBy={sortBy}
            sortOrder={sortOrder}
            onSort={handleSort}
          />
        )}
      </div>

      {/* Info Note */}
      <div className="card p-4 bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800">
        <p className="text-sm text-blue-800 dark:text-blue-200">
          ℹ️ For detailed study information, experimental protocols, and publication details, 
          please visit the{' '}
          <a
            href={generateGeoUrl(gseGroup.gseAccession)}
            target="_blank"
            rel="noopener noreferrer"
            className="font-medium underline hover:text-blue-900 dark:hover:text-blue-100"
          >
            NCBI GEO page
          </a>
          .
        </p>
      </div>
    </div>
  );
}