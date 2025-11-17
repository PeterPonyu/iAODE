'use client';

import { useState, useMemo } from 'react';
import { GSEGroup, FilterState } from '@/types/datasets';
import DatasetFilters from './DatasetFilters';
import DatasetSearch from './DatasetSearch';
import DatasetGrid from './DatasetGrid';
import DatasetTable from './DatasetTable';
import ViewToggle from './ViewToggle';
import CategoryLegend from '@/components/ui/CategoryLegend';
import DataTypeTabs from './DataTypeTabs';
import { applyFilters } from '@/lib/filterUtils';
import { searchGSE } from '@/lib/searchUtils';

interface DatasetBrowserProps {
  initialData: GSEGroup[];
  dataType: 'ATAC' | 'RNA';
  totalDatasets: number;
}

export default function DatasetBrowser({ initialData, dataType, totalDatasets }: DatasetBrowserProps) {
  const [viewMode, setViewMode] = useState<'grid' | 'table'>('grid');
  const [searchQuery, setSearchQuery] = useState('');
  const [filters, setFilters] = useState<FilterState>({
    search: '',
    categories: [],
    organisms: [],
    cellRange: null,
    featureRange: null,
  });

  // Apply search first (on GSE level)
  const searchedGSEGroups = useMemo(() => 
    searchGSE(initialData, searchQuery),
    [initialData, searchQuery]
  );

  // Then apply filters (on dataset level within each GSE)
  const filteredGSEGroups = useMemo(() => {
    return searchedGSEGroups
      .map(gse => ({
        ...gse,
        datasets: applyFilters(gse.datasets, filters)
      }))
      .filter(gse => gse.datasets.length > 0);
  }, [searchedGSEGroups, filters]);

  const filteredTotalDatasets = filteredGSEGroups.reduce(
    (sum, g) => sum + g.datasets.length, 
    0
  );

  // Feature label based on type
  const featureLabel = dataType === 'ATAC' ? 'peaks' : 'genes';

  return (
    <div className="space-y-6">
      {/* Data Type Tabs */}
      <DataTypeTabs currentType={dataType} />

      <div className="flex flex-col lg:flex-row gap-6">
        {/* Filters Sidebar */}
        <aside className="flex-shrink-0 lg:w-64">
          <div className="flex flex-col gap-4 lg:sticky lg:top-4">
            <DatasetFilters
              filters={filters}
              onFiltersChange={setFilters}
              allDatasets={initialData.flatMap(g => g.datasets)}
              dataType={dataType}
            />
            {/* Legend in sidebar for desktop */}
            <div className="hidden lg:block">
              <CategoryLegend compact={false} />
            </div>
          </div>
        </aside>

        {/* Main Content Area */}
        <div className="flex-1 min-w-0">
          {/* Search & Controls */}
          <div className="flex flex-col gap-4 mb-6">
            <DatasetSearch
              value={searchQuery}
              onChange={setSearchQuery}
              placeholder={`Search ${dataType === 'ATAC' ? 'scATAC-seq' : 'scRNA-seq'} datasets by GSE ID, title, authors, or organism...`}
            />
            
            {/* Compact Legend for mobile/tablet */}
            <div className="lg:hidden">
              <CategoryLegend compact showTitle />
            </div>
            
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
              <p className="text-sm text-[rgb(var(--text-secondary))] transition-colors">
                Showing <span className="font-semibold text-[rgb(var(--text-primary))] transition-colors">{filteredGSEGroups.length}</span> studies 
                ({filteredTotalDatasets} datasets) of {initialData.length} total
              </p>
              <ViewToggle mode={viewMode} onChange={setViewMode} />
            </div>
          </div>

          {/* Results */}
          {filteredGSEGroups.length === 0 ? (
            <div className="text-center py-12">
              <div className="text-[rgb(var(--text-muted))] mb-4 transition-colors">
                <svg className="mx-auto h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h3 className="text-lg font-medium text-[rgb(var(--text-primary))] mb-2 transition-colors">
                No datasets found
              </h3>
              <p className="text-[rgb(var(--text-tertiary))] mb-4 transition-colors">
                Try adjusting your search or filters
              </p>
              <button
                onClick={() => {
                  setSearchQuery('');
                  setFilters({
                    search: '',
                    categories: [],
                    organisms: [],
                    cellRange: null,
                    featureRange: null,
                  });
                }}
                className="btn-secondary"
              >
                Clear all filters
              </button>
            </div>
          ) : viewMode === 'grid' ? (
            <DatasetGrid data={filteredGSEGroups} dataType={dataType} />
          ) : (
            <DatasetTable data={filteredGSEGroups} dataType={dataType} />
          )}
        </div>
      </div>
    </div>
  );
}