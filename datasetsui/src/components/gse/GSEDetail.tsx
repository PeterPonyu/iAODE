'use client';

import { useState, useMemo } from 'react';
import Link from 'next/link';
import { GSEGroup } from '@/types/datasets';
import { generateGeoUrl } from '@/lib/geoUtils';
import GSEHeader from './GSEHeader';
import GSEStats from './GSEStats';
import DatasetList from './DatasetList';
import DatasetListTable from './DatasetListTable';
import { ArrowLeft, LayoutGrid, List } from 'lucide-react';
import { cn } from '@/lib/utils';

interface GSEDetailProps {
  gseGroup: GSEGroup;
  dataType: 'ATAC' | 'RNA';
}

export default function GSEDetail({ gseGroup, dataType }: GSEDetailProps) {
  const [viewMode, setViewMode] = useState<'card' | 'table'>('card');
  const [sortBy, setSortBy] = useState<'cells' | 'features' | 'size' | 'filename'>('cells');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  const featureLabel = dataType === 'ATAC' ? 'Peaks' : 'Genes';

  // Sort datasets
  const sortedDatasets = useMemo(() => {
    const sorted = [...gseGroup.datasets].sort((a, b) => {
      let aVal: number, bVal: number;

      switch (sortBy) {
        case 'cells':
          aVal = parseInt(a.nCells.toString()) || 0;
          bVal = parseInt(b.nCells.toString()) || 0;
          break;
        case 'features':
          aVal = parseInt(a.nFeatures.toString()) || 0;
          bVal = parseInt(b.nFeatures.toString()) || 0;
          break;
        case 'size':
          aVal = parseFloat(a.dataFileSize.toString()) || 0;
          bVal = parseFloat(b.dataFileSize.toString()) || 0;
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
        href={`/datasets?type=${dataType}`}
        className="inline-flex items-center gap-2 text-sm text-[rgb(var(--muted-foreground))] hover:text-[rgb(var(--foreground))] transition-colors"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to Browse
      </Link>

      {/* Header Section */}
      <GSEHeader gseGroup={gseGroup} dataType={dataType} />

      {/* Stats Cards */}
      <GSEStats gseGroup={gseGroup} dataType={dataType} />

      {/* Datasets Section */}
      <div className="card p-6">
        {/* Section Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
          <div>
            <h2 className="text-xl font-bold text-[rgb(var(--foreground))] mb-1 transition-colors">
              Datasets in this Study
            </h2>
            <p className="text-sm text-[rgb(var(--muted-foreground))] transition-colors">
              {gseGroup.datasets.length} dataset{gseGroup.datasets.length !== 1 ? 's' : ''} available in 10X h5 format
            </p>
          </div>

          {/* Controls */}
          <div className="flex items-center gap-3">
            {/* Sort Dropdown */}
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
              className="px-3 py-2 text-sm border border-[rgb(var(--border))] rounded-lg bg-[rgb(var(--card))] text-[rgb(var(--foreground))] focus:outline-none focus:ring-2 focus:ring-[rgb(var(--primary))] transition-colors"
            >
              <option value="cells">Sort by Cells</option>
              <option value="features">Sort by {featureLabel}</option>
              <option value="size">Sort by Size</option>
              <option value="filename">Sort by Filename</option>
            </select>

            {/* View Toggle */}
            <div className="inline-flex rounded-lg border border-[rgb(var(--border))] bg-[rgb(var(--card))] p-1 transition-colors">
              <button
                onClick={() => setViewMode('card')}
                className={cn(
                  'inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
                  viewMode === 'card'
                    ? 'bg-[rgb(var(--primary))] text-white'
                    : 'text-[rgb(var(--text-secondary))] hover:bg-[rgb(var(--muted))]'
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
                    ? 'bg-[rgb(var(--primary))] text-white'
                    : 'text-[rgb(var(--text-secondary))] hover:bg-[rgb(var(--muted))]'
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
          <DatasetList datasets={sortedDatasets} dataType={dataType} />
        ) : (
          <DatasetListTable 
            datasets={sortedDatasets} 
            dataType={dataType}
            sortBy={sortBy}
            sortOrder={sortOrder}
            onSort={handleSort}
          />
        )}
      </div>

      {/* Info Note */}
      <div className="card p-4 bg-[rgb(var(--info-bg))] border-[rgb(var(--info-border))] transition-colors">
        <p className="text-sm text-[rgb(var(--info-text))] transition-colors">
          ℹ️ All datasets are provided in standardized 10X Genomics HDF5 filtered matrix format. 
          For detailed study information, experimental protocols, and publication details, 
          please visit the{' '}
          <a
            href={generateGeoUrl(gseGroup.gseAccession)}
            target="_blank"
            rel="noopener noreferrer"
            className="font-medium underline hover:text-[rgb(var(--info-text-strong))] transition-colors"
          >
            NCBI GEO page
          </a>
          .
        </p>
      </div>
    </div>
  );
}